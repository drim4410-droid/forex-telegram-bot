import os
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime

import httpx
import aiosqlite

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode


# ================== ENV ==================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "").strip()
ADMIN_ID = int(os.getenv("ADMIN_ID", "0").strip() or 0)

if not BOT_TOKEN:
    raise RuntimeError("ENV BOT_TOKEN is missing")
if not TWELVE_API_KEY:
    raise RuntimeError("ENV TWELVE_API_KEY is missing")
if not ADMIN_ID:
    raise RuntimeError("ENV ADMIN_ID is missing (set your Telegram user id)")

DB_PATH = "bot.db"

SUPPORTED_SYMBOLS = ["EUR/USD", "XAU/USD"]
SUPPORTED_TF = ["5min", "15min", "30min"]
TF_LABELS = {"5min": "5M", "15min": "15M", "30min": "30M"}

CANDLES = 120
TP_SL_CHECK_EVERY = 30  # секунд
ACCESS_DAYS = 30
ACCESS_SECONDS = ACCESS_DAYS * 24 * 60 * 60

AUTO_INTERVALS_MIN = [5, 15, 30]  # доступные интервалы авто-анализа


# ================== DATA ==================
@dataclass
class Signal:
    user_id: int
    symbol: str
    tf: str
    direction: str  # BUY/SELL
    entry: float
    tp: float
    sl: float
    created_at: int
    is_active: int = 1


def now_ts() -> int:
    return int(datetime.utcnow().timestamp())


def utc_fmt(ts: int) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M UTC")


# ================== INDICATORS ==================
def ema(values, period: int):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema_val = sum(values[:period]) / period
    for v in values[period:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val


def rsi(values, period: int = 14):
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses += abs(diff)
    avg_gain = gains / period
    avg_loss = losses / period

    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = max(diff, 0)
        loss = max(-diff, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(highs, lows, closes, period: int = 14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    atr_val = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return atr_val


def fmt_price(symbol: str, price: float) -> str:
    if symbol == "EUR/USD":
        return f"{price:.5f}"
    return f"{price:.2f}"


# ================== API ==================
async def fetch_candles(symbol: str, interval: str):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": str(CANDLES),
        "apikey": TWELVE_API_KEY,
        "format": "JSON",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params)
        data = r.json()

    if data.get("status") == "error":
        raise RuntimeError(data.get("message", "TwelveData error"))

    values = data.get("values", [])
    if not values or len(values) < 30:
        return None

    values = list(reversed(values))  # от старых к новым
    highs = [float(v["high"]) for v in values]
    lows = [float(v["low"]) for v in values]
    closes = [float(v["close"]) for v in values]
    return highs, lows, closes


async def fetch_quote(symbol: str) -> float | None:
    url = "https://api.twelvedata.com/quote"
    params = {"symbol": symbol, "apikey": TWELVE_API_KEY, "format": "JSON"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        data = r.json()

    if data.get("status") == "error":
        return None

    try:
        return float(data["price"])
    except Exception:
        return None


# ================== STRATEGY ==================
def make_signal(symbol: str, tf: str, highs, lows, closes):
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    r = rsi(closes, 14)
    a = atr(highs, lows, closes, 14)

    if ema9 is None or ema21 is None or r is None or a is None:
        return None

    last = closes[-1]

    direction = None
    if ema9 > ema21 and r >= 55:
        direction = "BUY"
    elif ema9 < ema21 and r <= 45:
        direction = "SELL"
    else:
        return None

    tp_mult = 1.2
    sl_mult = 0.8

    entry = last
    if direction == "BUY":
        tp = entry + a * tp_mult
        sl = entry - a * sl_mult
    else:
        tp = entry - a * tp_mult
        sl = entry + a * sl_mult

    note = f"EMA9 {'>' if direction=='BUY' else '<'} EMA21 | RSI={r:.1f} | ATR={a:.6f}"
    return direction, entry, tp, sl, note


# ================== DB ==================
async def db_init():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY
        )
        """)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            tf TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry REAL NOT NULL,
            tp REAL NOT NULL,
            sl REAL NOT NULL,
            created_at INTEGER NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        )
        """)
        await db.commit()

        # миграция users -> доступ
        cur = await db.execute("PRAGMA table_info(users)")
        cols = [row[1] for row in await cur.fetchall()]

        if "status" not in cols:
            await db.execute("ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'")
        if "approved_until" not in cols:
            await db.execute("ALTER TABLE users ADD COLUMN approved_until INTEGER NOT NULL DEFAULT 0")
        if "requested_at" not in cols:
            await db.execute("ALTER TABLE users ADD COLUMN requested_at INTEGER NOT NULL DEFAULT 0")

        await db.commit()

        # настройки авто-анализа
        await db.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            auto_enabled INTEGER NOT NULL DEFAULT 0,
            auto_interval_min INTEGER NOT NULL DEFAULT 15,
            auto_symbols TEXT NOT NULL DEFAULT '["EUR/USD","XAU/USD"]'
        )
        """)
        await db.execute("""
        CREATE TABLE IF NOT EXISTS auto_state (
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            tf TEXT NOT NULL,
            last_fingerprint TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (user_id, symbol, tf)
        )
        """)
        await db.commit()


async def ensure_user(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        if not await cur.fetchone():
            await db.execute(
                "INSERT INTO users (user_id, status, approved_until, requested_at) VALUES (?, 'pending', 0, 0)",
                (user_id,)
            )
        cur2 = await db.execute("SELECT user_id FROM user_settings WHERE user_id=?", (user_id,))
        if not await cur2.fetchone():
            await db.execute("INSERT INTO user_settings (user_id) VALUES (?)", (user_id,))
        await db.commit()


async def get_user_access(user_id: int) -> tuple[str, int, int]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT status, approved_until, requested_at FROM users WHERE user_id=?",
            (user_id,)
        )
        row = await cur.fetchone()
        if not row:
            return "pending", 0, 0
        return row[0], int(row[1]), int(row[2])


async def set_user_pending(user_id: int):
    ts = now_ts()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET status='pending', requested_at=? WHERE user_id=?",
            (ts, user_id)
        )
        await db.commit()


async def approve_user_30d(user_id: int):
    ts = now_ts()
    until = ts + ACCESS_SECONDS
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET status='approved', approved_until=?, requested_at=0 WHERE user_id=?",
            (until, user_id)
        )
        await db.commit()
    return until


async def extend_user_30d(user_id: int):
    status, approved_until, _ = await get_user_access(user_id)
    base = approved_until if (status == "approved" and approved_until > now_ts()) else now_ts()
    until = base + ACCESS_SECONDS
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET status='approved', approved_until=? WHERE user_id=?",
            (until, user_id)
        )
        await db.commit()
    return until


async def block_user(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET status='blocked', approved_until=0 WHERE user_id=?",
            (user_id,)
        )
        await db.commit()


def is_access_active(status: str, approved_until: int) -> bool:
    return status == "approved" and approved_until > now_ts()


async def get_active_signal(user_id: int) -> Signal | None:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            SELECT user_id, symbol, tf, direction, entry, tp, sl, created_at, is_active
            FROM signals
            WHERE user_id=? AND is_active=1
            ORDER BY id DESC
            LIMIT 1
        """, (user_id,))
        row = await cur.fetchone()
        if not row:
            return None
        return Signal(*row)


async def create_signal(sig: Signal):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO signals (user_id, symbol, tf, direction, entry, tp, sl, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (sig.user_id, sig.symbol, sig.tf, sig.direction, sig.entry, sig.tp, sig.sl, sig.created_at))
        await db.commit()


async def close_signal(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE signals SET is_active=0 WHERE user_id=? AND is_active=1", (user_id,))
        await db.commit()


# ---------- settings ----------
async def get_settings(user_id: int) -> tuple[int, int, list[str]]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT auto_enabled, auto_interval_min, auto_symbols FROM user_settings WHERE user_id=?",
            (user_id,)
        )
        row = await cur.fetchone()
        if not row:
            return 0, 15, SUPPORTED_SYMBOLS[:]
        auto_enabled = int(row[0])
        interval = int(row[1])
        try:
            symbols = json.loads(row[2]) if row[2] else SUPPORTED_SYMBOLS[:]
        except Exception:
            symbols = SUPPORTED_SYMBOLS[:]
        symbols = [s for s in symbols if s in SUPPORTED_SYMBOLS]
        if not symbols:
            symbols = SUPPORTED_SYMBOLS[:]
        if interval not in AUTO_INTERVALS_MIN:
            interval = 15
        return auto_enabled, interval, symbols


async def set_auto_enabled(user_id: int, enabled: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE user_settings SET auto_enabled=? WHERE user_id=?", (int(enabled), user_id))
        await db.commit()


async def set_auto_interval(user_id: int, interval_min: int):
    if interval_min not in AUTO_INTERVALS_MIN:
        interval_min = 15
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE user_settings SET auto_interval_min=? WHERE user_id=?", (interval_min, user_id))
        await db.commit()


async def toggle_symbol(user_id: int, symbol: str):
    if symbol not in SUPPORTED_SYMBOLS:
        return
    enabled, interval, symbols = await get_settings(user_id)
    if symbol in symbols and len(symbols) > 1:
        symbols.remove(symbol)
    elif symbol not in symbols:
        symbols.append(symbol)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE user_settings SET auto_symbols=? WHERE user_id=?",
            (json.dumps(symbols, ensure_ascii=False), user_id)
        )
        await db.commit()


async def get_approved_users() -> list[tuple[int, int]]:
    """returns list of (user_id, approved_until) for active approved users"""
    ts = now_ts()
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT user_id, approved_until FROM users WHERE status='approved' AND approved_until > ?",
            (ts,)
        )
        return [(int(r[0]), int(r[1])) for r in await cur.fetchall()]


async def get_pending_users(limit: int = 10) -> list[int]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT user_id FROM users WHERE status='pending' AND requested_at > 0 ORDER BY requested_at DESC LIMIT ?",
            (limit,)
        )
        return [int(r[0]) for r in await cur.fetchall()]


async def set_last_fingerprint(user_id: int, symbol: str, tf: str, fp: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO auto_state (user_id, symbol, tf, last_fingerprint)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, symbol, tf) DO UPDATE SET last_fingerprint=excluded.last_fingerprint
        """, (user_id, symbol, tf, fp))
        await db.commit()


async def get_last_fingerprint(user_id: int, symbol: str, tf: str) -> str:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT last_fingerprint FROM auto_state WHERE user_id=? AND symbol=? AND tf=?",
            (user_id, symbol, tf)
        )
        row = await cur.fetchone()
        return row[0] if row else ""


# ================== UI ==================
def main_kb(is_admin: bool = False):
    kb = ReplyKeyboardBuilder()
    kb.button(text="📍 Новый сигнал")
    kb.button(text="ℹ️ Помощь")
    kb.button(text="⚙️ Настройки")
    if is_admin:
        kb.button(text="🛠 Админ")
    kb.adjust(2)
    return kb.as_markup(resize_keyboard=True)


def locked_kb(is_admin: bool = False):
    kb = ReplyKeyboardBuilder()
    kb.button(text="🔐 Запросить доступ")
    kb.button(text="ℹ️ Помощь")
    if is_admin:
        kb.button(text="🛠 Админ")
    kb.adjust(2)
    return kb.as_markup(resize_keyboard=True)


def admin_req_kb(user_id: int):
    ikb = InlineKeyboardBuilder()
    ikb.button(text="✅ Одобрить на 30 дней", callback_data=f"approve:{user_id}")
    ikb.button(text="➕ Продлить +30 дней", callback_data=f"extend:{user_id}")
    ikb.button(text="⛔️ Заблокировать", callback_data=f"block:{user_id}")
    ikb.adjust(1)
    return ikb.as_markup()


def settings_kb(user_id: int, enabled: int, interval: int, symbols: list[str]):
    ikb = InlineKeyboardBuilder()
    ikb.button(text=f"🔁 Авто-анализ: {'ВКЛ' if enabled else 'ВЫКЛ'}", callback_data=f"set:auto:{1 if not enabled else 0}")
    # интервалы
    for m in AUTO_INTERVALS_MIN:
        mark = "✅" if m == interval else "▫️"
        ikb.button(text=f"{mark} ⏱ {m} мин", callback_data=f"set:int:{m}")
    # пары
    for s in SUPPORTED_SYMBOLS:
        mark = "✅" if s in symbols else "▫️"
        ikb.button(text=f"{mark} 📌 {s}", callback_data=f"set:sym:{s}")
    ikb.button(text="⬅️ Закрыть", callback_data="set:close")
    ikb.adjust(1, 3, 2, 1)
    return ikb.as_markup()


def admin_panel_kb():
    ikb = InlineKeyboardBuilder()
    ikb.button(text="🟡 Заявки", callback_data="adm:pending")
    ikb.button(text="✅ Активные", callback_data="adm:active")
    ikb.button(text="📣 Рассылка", callback_data="adm:bcast")
    ikb.button(text="⬅️ Закрыть", callback_data="adm:close")
    ikb.adjust(2, 1, 1)
    return ikb.as_markup()


# ================== BOT ==================
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()

watch_tasks: dict[int, asyncio.Task] = {}
auto_task: asyncio.Task | None = None


def signal_text_common(symbol: str, tf: str, direction: str, entry: float, tp: float, sl: float, note: str | None):
    emoji = "🟢" if direction == "BUY" else "🔴"
    return (
        f"📊 <b>{symbol} SIGNAL</b> <i>({TF_LABELS.get(tf, tf)})</i>\n\n"
        f"<b>Direction:</b> {emoji} <b>{direction}</b>\n"
        f"<b>Entry:</b> <code>{fmt_price(symbol, entry)}</code>\n"
        f"<b>Take Profit:</b> <code>{fmt_price(symbol, tp)}</code>\n"
        f"<b>Stop Loss:</b> <code>{fmt_price(symbol, sl)}</code>\n"
        + (f"\n<b>Note:</b> {note}\n" if note else "\n")
        + "\n⚠️ <i>Не является финансовой рекомендацией.</i>"
    )


def signal_text(sig: Signal, note: str | None = None) -> str:
    return signal_text_common(sig.symbol, sig.tf, sig.direction, sig.entry, sig.tp, sig.sl, note)


async def require_access_or_lock(m: Message) -> bool:
    await ensure_user(m.from_user.id)
    status, approved_until, _ = await get_user_access(m.from_user.id)

    if status == "approved" and approved_until <= now_ts():
        await set_user_pending(m.from_user.id)
        status = "pending"
        approved_until = 0

    if status == "blocked":
        await m.answer("⛔️ Доступ заблокирован. Напиши администратору.")
        return False

    if is_access_active(status, approved_until):
        return True

    await m.answer(
        "🔒 Доступ к сигналам закрыт.\n\n"
        "Нажми <b>🔐 Запросить доступ</b>, и после моего одобрения у тебя будет доступ на <b>30 дней</b>.",
        reply_markup=locked_kb(is_admin=(m.from_user.id == ADMIN_ID))
    )
    return False


async def start_watch(user_id: int):
    if user_id in watch_tasks and not watch_tasks[user_id].done():
        return

    async def _loop():
        while True:
            sig = await get_active_signal(user_id)
            if not sig:
                return

            price = await fetch_quote(sig.symbol)
            if price is None:
                await asyncio.sleep(TP_SL_CHECK_EVERY)
                continue

            hit_tp = False
            hit_sl = False

            if sig.direction == "BUY":
                if price >= sig.tp:
                    hit_tp = True
                elif price <= sig.sl:
                    hit_sl = True
            else:
                if price <= sig.tp:
                    hit_tp = True
                elif price >= sig.sl:
                    hit_sl = True

            if hit_tp or hit_sl:
                await close_signal(user_id)
                if hit_tp:
                    await bot.send_message(
                        user_id,
                        f"✅ <b>TP достигнут!</b>\n"
                        f"{sig.symbol} {TF_LABELS.get(sig.tf)} {sig.direction}\n"
                        f"Цена: <code>{fmt_price(sig.symbol, price)}</code>\n\n"
                        f"Теперь можно нажать <b>📍 Новый сигнал</b>."
                    )
                else:
                    await bot.send_message(
                        user_id,
                        f"❌ <b>SL сработал</b>\n"
                        f"{sig.symbol} {TF_LABELS.get(sig.tf)} {sig.direction}\n"
                        f"Цена: <code>{fmt_price(sig.symbol, price)}</code>\n\n"
                        f"Теперь можно нажать <b>📍 Новый сигнал</b>."
                    )
                return

            await asyncio.sleep(TP_SL_CHECK_EVERY)

    watch_tasks[user_id] = asyncio.create_task(_loop())


# ================== AUTO ANALYSIS ==================
def fingerprint(symbol: str, tf: str, direction: str, entry: float, tp: float, sl: float) -> str:
    # грубо округляем, чтобы мелкий шум не считался новым сигналом
    e = fmt_price(symbol, entry)
    t = fmt_price(symbol, tp)
    s = fmt_price(symbol, sl)
    return f"{symbol}|{tf}|{direction}|{e}|{t}|{s}"


async def auto_loop():
    # общий цикл: каждый 60 сек проверяем, кому пора отправлять
    while True:
        try:
            users = await get_approved_users()
            ts = now_ts()

            for user_id, _until in users:
                await ensure_user(user_id)
                enabled, interval_min, symbols = await get_settings(user_id)
                if not enabled:
                    continue

                # чтобы не делать отдельный last_run в БД — используем простую задержку:
                # на каждый цикл проверяем и отправляем, но антиспам по fingerprint не даст флудить.
                # Однако, чтобы уменьшить запросы, делаем "пакетную" проверку по интервалам:
                if (ts // 60) % interval_min != 0:
                    continue

                # ищем сигнал (первый сильный)
                for symbol in symbols:
                    for tf in SUPPORTED_TF:
                        try:
                            candles = await fetch_candles(symbol, tf)
                            if not candles:
                                continue
                            highs, lows, closes = candles
                            res = make_signal(symbol, tf, highs, lows, closes)
                            if not res:
                                continue
                            direction, entry, tp, sl, note = res

                            fp = fingerprint(symbol, tf, direction, float(entry), float(tp), float(sl))
                            last_fp = await get_last_fingerprint(user_id, symbol, tf)
                            if fp == last_fp:
                                continue  # уже отправляли такое

                            await set_last_fingerprint(user_id, symbol, tf, fp)

                            text = (
                                "🤖 <b>Авто-анализ</b>\n\n" +
                                signal_text_common(symbol, tf, direction, float(entry), float(tp), float(sl), note)
                            )
                            await bot.send_message(user_id, text, reply_markup=main_kb(is_admin=(user_id == ADMIN_ID)))
                            # чуть притормозим, чтобы не словить лимиты
                            await asyncio.sleep(0.8)
                        except Exception:
                            continue

        except Exception:
            # чтобы цикл не падал
            pass

        await asyncio.sleep(60)


# ================== HANDLERS ==================
@dp.message(Command("start"))
async def start_cmd(m: Message):
    await ensure_user(m.from_user.id)
    status, approved_until, _ = await get_user_access(m.from_user.id)

    if status == "approved" and approved_until <= now_ts():
        await set_user_pending(m.from_user.id)
        status = "pending"
        approved_until = 0

    is_admin = (m.from_user.id == ADMIN_ID)

    if is_access_active(status, approved_until):
        await m.answer(
            "Привет! Я выдаю сигналы по <b>EUR/USD</b> и <b>XAU/USD</b>.\n\n"
            "• <b>📍 Новый сигнал</b> — ручной сигнал\n"
            "• <b>⚙️ Настройки</b> — авто-анализ\n",
            reply_markup=main_kb(is_admin=is_admin)
        )
        await start_watch(m.from_user.id)
        return

    await m.answer(
        "Привет! Чтобы пользоваться ботом, нужен доступ.\n\n"
        "Нажми <b>🔐 Запросить доступ</b>. После моего одобрения доступ будет активен <b>30 дней</b>.",
        reply_markup=locked_kb(is_admin=is_admin)
    )


@dp.message(F.text == "🔐 Запросить доступ")
async def request_access(m: Message):
    await ensure_user(m.from_user.id)

    status, approved_until, requested_at = await get_user_access(m.from_user.id)

    if is_access_active(status, approved_until):
        await m.answer("✅ У тебя уже есть активный доступ.", reply_markup=main_kb(is_admin=(m.from_user.id == ADMIN_ID)))
        return

    if status == "blocked":
        await m.answer("⛔️ Доступ заблокирован. Напиши администратору.")
        return

    ts = now_ts()
    if requested_at and (ts - requested_at) < 120:
        await m.answer("⏳ Заявка уже отправлена. Подожди немного — я отвечу.", reply_markup=locked_kb(is_admin=(m.from_user.id == ADMIN_ID)))
        return

    await set_user_pending(m.from_user.id)

    username = f"@{m.from_user.username}" if m.from_user.username else "без username"
    await m.answer("✅ Заявка отправлена. Как только я одобрю — доступ включится на 30 дней.", reply_markup=locked_kb(is_admin=(m.from_user.id == ADMIN_ID)))

    try:
        await bot.send_message(
            ADMIN_ID,
            "🔔 <b>Запрос доступа</b>\n\n"
            f"User ID: <code>{m.from_user.id}</code>\n"
            f"Username: {username}\n"
            f"Name: {m.from_user.full_name}",
            reply_markup=admin_req_kb(m.from_user.id)
        )
    except Exception:
        pass


@dp.message(F.text == "ℹ️ Помощь")
async def help_(m: Message):
    await ensure_user(m.from_user.id)
    status, approved_until, _ = await get_user_access(m.from_user.id)

    if status == "approved" and approved_until <= now_ts():
        await set_user_pending(m.from_user.id)
        status = "pending"
        approved_until = 0

    if is_access_active(status, approved_until):
        await m.answer(
            "ℹ️ <b>Помощь</b>\n\n"
            "• <b>📍 Новый сигнал</b> — ручной сигнал (1 активный)\n"
            "• <b>⚙️ Настройки</b> — авто-анализ и интервалы\n\n"
            "⚠️ Сигналы не гарантия прибыли."
        )
    else:
        await m.answer(
            "ℹ️ <b>Помощь</b>\n\n"
            "Чтобы пользоваться сигналами, нужен доступ от администратора.\n"
            "Нажми <b>🔐 Запросить доступ</b>.\n"
            f"После одобрения доступ действует <b>{ACCESS_DAYS} дней</b>.",
            reply_markup=locked_kb(is_admin=(m.from_user.id == ADMIN_ID))
        )


@dp.message(F.text == "⚙️ Настройки")
async def settings_open(m: Message):
    if not await require_access_or_lock(m):
        return
    enabled, interval, symbols = await get_settings(m.from_user.id)
    await m.answer(
        "⚙️ <b>Настройки авто-анализа</b>\n\n"
        "Включи авто-анализ, выбери интервал и пары.",
        reply_markup=settings_kb(m.from_user.id, enabled, interval, symbols)
    )


@dp.callback_query(F.data.startswith("set:"))
async def settings_cb(q: CallbackQuery):
    await q.answer()
    user_id = q.from_user.id

    # закрыть
    if q.data == "set:close":
        try:
            await q.message.delete()
        except Exception:
            pass
        return

    # доступ нужен (кроме админа — но админ тоже должен быть approved чтобы настройки работали)
    await ensure_user(user_id)
    status, approved_until, _ = await get_user_access(user_id)
    if not is_access_active(status, approved_until):
        await q.message.edit_text("🔒 Доступ не активен. Запроси доступ у администратора.")
        return

    parts = q.data.split(":")
    if len(parts) < 3:
        return

    kind = parts[1]
    val = ":".join(parts[2:])

    if kind == "auto":
        await set_auto_enabled(user_id, int(val))
    elif kind == "int":
        try:
            await set_auto_interval(user_id, int(val))
        except Exception:
            pass
    elif kind == "sym":
        await toggle_symbol(user_id, val)

    enabled, interval, symbols = await get_settings(user_id)
    await q.message.edit_reply_markup(reply_markup=settings_kb(user_id, enabled, interval, symbols))


@dp.message(F.text == "📍 Новый сигнал")
async def new_signal(m: Message):
    if not await require_access_or_lock(m):
        return

    active = await get_active_signal(m.from_user.id)
    if active:
        await m.answer(
            "⛔️ Уже есть активный сигнал.\n"
            "Новый появится после TP/SL (я уведомлю автоматически)."
        )
        await start_watch(m.from_user.id)
        return

    best = None
    best_note = None

    for symbol in SUPPORTED_SYMBOLS:
        for tf in SUPPORTED_TF:
            try:
                candles = await fetch_candles(symbol, tf)
                if not candles:
                    continue
                highs, lows, closes = candles
                res = make_signal(symbol, tf, highs, lows, closes)
                if not res:
                    continue
                direction, entry, tp, sl, note = res
                best = (symbol, tf, direction, entry, tp, sl)
                best_note = note
                break
            except Exception:
                continue
        if best:
            break

    if not best:
        await m.answer("Сейчас нет достаточно сильного сигнала. Попробуй позже.")
        return

    symbol, tf, direction, entry, tp, sl = best

    sig = Signal(
        user_id=m.from_user.id,
        symbol=symbol,
        tf=tf,
        direction=direction,
        entry=float(entry),
        tp=float(tp),
        sl=float(sl),
        created_at=now_ts(),
        is_active=1
    )
    await create_signal(sig)

    await m.answer("✅ Сигнал найден. Я отслеживаю TP/SL и уведомлю автоматически.")
    await m.answer(signal_text(sig, note=best_note), reply_markup=main_kb(is_admin=(m.from_user.id == ADMIN_ID)))
    await start_watch(m.from_user.id)


# ================== ADMIN ==================
@dp.message(Command("admin"))
async def admin_cmd(m: Message):
    if m.from_user.id != ADMIN_ID:
        return
    await m.answer("🛠 <b>Админ-панель</b>", reply_markup=admin_panel_kb())


@dp.message(F.text == "🛠 Админ")
async def admin_btn(m: Message):
    if m.from_user.id != ADMIN_ID:
        return
    await m.answer("🛠 <b>Админ-панель</b>", reply_markup=admin_panel_kb())


@dp.callback_query(F.data.startswith("adm:"))
async def admin_panel_cb(q: CallbackQuery):
    if q.from_user.id != ADMIN_ID:
        await q.answer("Нет прав.", show_alert=True)
        return
    await q.answer()

    if q.data == "adm:close":
        try:
            await q.message.delete()
        except Exception:
            pass
        return

    if q.data == "adm:pending":
        pending = await get_pending_users(limit=10)
        if not pending:
            await q.message.edit_text("🟡 Заявок нет.", reply_markup=admin_panel_kb())
            return

        text = "🟡 <b>Заявки (последние 10)</b>\n\n" + "\n".join([f"• <code>{uid}</code>" for uid in pending])
        await q.message.edit_text(text, reply_markup=admin_panel_kb())
        # отдельно кидаем карточки с кнопками
        for uid in pending:
            try:
                await bot.send_message(ADMIN_ID, f"Заявка: <code>{uid}</code>", reply_markup=admin_req_kb(uid))
            except Exception:
                pass
        return

    if q.data == "adm:active":
        users = await get_approved_users()
        if not users:
            await q.message.edit_text("✅ Активных нет.", reply_markup=admin_panel_kb())
            return
        lines = []
        for uid, until in users[:20]:
            enabled, interval, symbols = await get_settings(uid)
            lines.append(f"• <code>{uid}</code> до <code>{utc_fmt(until)}</code> | авто={'ON' if enabled else 'OFF'} | {interval}m")
        text = "✅ <b>Активные (до 20)</b>\n\n" + "\n".join(lines) + "\n\nЧтобы управлять — открой карточку пользователя по user_id через заявки/сообщение."
        await q.message.edit_text(text, reply_markup=admin_panel_kb())
        return

    if q.data == "adm:bcast":
        await q.message.edit_text(
            "📣 <b>Рассылка</b>\n\n"
            "Отправь мне сообщение в чат и начни его с:\n"
            "<code>/broadcast</code> пробел текст\n\n"
            "Пример:\n"
            "<code>/broadcast Привет! Обновил авто-анализ.</code>",
            reply_markup=admin_panel_kb()
        )
        return


@dp.message(F.text.startswith("/broadcast"))
async def broadcast(m: Message):
    if m.from_user.id != ADMIN_ID:
        return
    parts = m.text.split(maxsplit=1)
    if len(parts) < 2:
        await m.answer("Напиши так: <code>/broadcast Текст сообщения</code>")
        return
    text = parts[1].strip()
    users = await get_approved_users()
    sent = 0
    for uid, _until in users:
        try:
            await bot.send_message(uid, "📣 <b>Сообщение</b>\n\n" + text)
            sent += 1
            await asyncio.sleep(0.5)
        except Exception:
            continue
    await m.answer(f"Готово. Отправлено: <b>{sent}</b>.")


@dp.callback_query(F.data.startswith("approve:"))
async def cb_approve(q: CallbackQuery):
    if q.from_user.id != ADMIN_ID:
        await q.answer("Недостаточно прав.", show_alert=True)
        return
    await q.answer()

    try:
        user_id = int(q.data.split("approve:")[1])
    except Exception:
        await q.answer("Ошибка данных.", show_alert=True)
        return

    await ensure_user(user_id)
    until = await approve_user_30d(user_id)

    await q.message.edit_text(q.message.text + f"\n\n✅ <b>Одобрено</b>\nДо: <code>{utc_fmt(until)}</code>")

    try:
        await bot.send_message(
            user_id,
            "✅ Доступ активирован!\n\n"
            f"Доступ до: <code>{utc_fmt(until)}</code>\n"
            "Нажми <b>📍 Новый сигнал</b> или включи авто-анализ в <b>⚙️ Настройки</b>.",
            reply_markup=main_kb(is_admin=(user_id == ADMIN_ID))
        )
    except Exception:
        pass


@dp.callback_query(F.data.startswith("extend:"))
async def cb_extend(q: CallbackQuery):
    if q.from_user.id != ADMIN_ID:
        await q.answer("Недостаточно прав.", show_alert=True)
        return
    await q.answer()

    try:
        user_id = int(q.data.split("extend:")[1])
    except Exception:
        await q.answer("Ошибка данных.", show_alert=True)
        return

    await ensure_user(user_id)
    until = await extend_user_30d(user_id)

    await q.message.edit_text(q.message.text + f"\n\n➕ <b>Продлено +30 дней</b>\nДо: <code>{utc_fmt(until)}</code>")

    try:
        await bot.send_message(
            user_id,
            f"➕ Доступ продлён.\nДо: <code>{utc_fmt(until)}</code>."
        )
    except Exception:
        pass


@dp.callback_query(F.data.startswith("block:"))
async def cb_block(q: CallbackQuery):
    if q.from_user.id != ADMIN_ID:
        await q.answer("Недостаточно прав.", show_alert=True)
        return
    await q.answer()

    try:
        user_id = int(q.data.split("block:")[1])
    except Exception:
        await q.answer("Ошибка данных.", show_alert=True)
        return

    await ensure_user(user_id)
    await block_user(user_id)

    await q.message.edit_text(q.message.text + "\n\n⛔️ <b>Пользователь заблокирован</b>")

    try:
        await bot.send_message(
            user_id,
            "⛔️ Доступ заблокирован. Если это ошибка — напиши администратору."
        )
    except Exception:
        pass


async def main():
    global auto_task
    await db_init()
    # запуск авто-анализа
    auto_task = asyncio.create_task(auto_loop())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
