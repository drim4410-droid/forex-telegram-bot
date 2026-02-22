import os
import asyncio
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

        # миграция users -> добавляем поля для доступа, если их нет
        cur = await db.execute("PRAGMA table_info(users)")
        cols = [row[1] for row in await cur.fetchall()]

        if "status" not in cols:
            await db.execute("ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'")
        if "approved_until" not in cols:
            await db.execute("ALTER TABLE users ADD COLUMN approved_until INTEGER NOT NULL DEFAULT 0")
        if "requested_at" not in cols:
            await db.execute("ALTER TABLE users ADD COLUMN requested_at INTEGER NOT NULL DEFAULT 0")

        await db.commit()


async def ensure_user(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        if not await cur.fetchone():
            await db.execute(
                "INSERT INTO users (user_id, status, approved_until, requested_at) VALUES (?, 'pending', 0, 0)",
                (user_id,)
            )
            await db.commit()


async def get_user_access(user_id: int) -> tuple[str, int, int]:
    """
    returns: (status, approved_until, requested_at)
    status: pending / approved / blocked
    """
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


# ================== UI ==================
def main_kb():
    kb = ReplyKeyboardBuilder()
    kb.button(text="📍 Новый сигнал")
    kb.button(text="ℹ️ Помощь")
    kb.adjust(2)
    return kb.as_markup(resize_keyboard=True)


def locked_kb():
    kb = ReplyKeyboardBuilder()
    kb.button(text="🔐 Запросить доступ")
    kb.button(text="ℹ️ Помощь")
    kb.adjust(2)
    return kb.as_markup(resize_keyboard=True)


def admin_approve_kb(user_id: int):
    ikb = InlineKeyboardBuilder()
    ikb.button(text="✅ Одобрить на 30 дней", callback_data=f"approve:{user_id}")
    ikb.button(text="⛔️ Заблокировать", callback_data=f"block:{user_id}")
    ikb.adjust(1)
    return ikb.as_markup()


# ================== BOT ==================
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()

watch_tasks: dict[int, asyncio.Task] = {}


def signal_text(sig: Signal, note: str | None = None) -> str:
    emoji = "🟢" if sig.direction == "BUY" else "🔴"
    return (
        f"📈 <b>{sig.symbol} SIGNAL</b> <i>({TF_LABELS.get(sig.tf, sig.tf)})</i>\n\n"
        f"<b>Direction:</b> {emoji} <b>{sig.direction}</b>\n"
        f"<b>Entry:</b> <code>{fmt_price(sig.symbol, sig.entry)}</code>\n"
        f"<b>Take Profit:</b> <code>{fmt_price(sig.symbol, sig.tp)}</code>\n"
        f"<b>Stop Loss:</b> <code>{fmt_price(sig.symbol, sig.sl)}</code>\n"
        + (f"\n<b>Note:</b> {note}\n" if note else "\n")
        + "\n⚠️ <i>Не является финансовой рекомендацией.</i>"
    )


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
        reply_markup=locked_kb()
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


@dp.message(Command("start"))
async def start_cmd(m: Message):
    await ensure_user(m.from_user.id)
    status, approved_until, _ = await get_user_access(m.from_user.id)

    if status == "approved" and approved_until <= now_ts():
        await set_user_pending(m.from_user.id)
        status = "pending"
        approved_until = 0

    if is_access_active(status, approved_until):
        await m.answer(
            "Привет! Я выдаю сигналы по <b>EUR/USD</b> и <b>XAU/USD</b>.\n\n"
            "Правила:\n"
            "• Нажми <b>📍 Новый сигнал</b> — получишь сигнал.\n"
            "• Пока сигнал активен — новый не выдаётся.\n"
            "• Я сам уведомлю, когда цена достигнет <b>TP</b> или <b>SL</b>.\n",
            reply_markup=main_kb()
        )
        await start_watch(m.from_user.id)
        return

    await m.answer(
        "Привет! Чтобы пользоваться ботом, нужен доступ.\n\n"
        "Нажми <b>🔐 Запросить доступ</b>. После моего одобрения доступ будет активен <b>30 дней</b>.",
        reply_markup=locked_kb()
    )


@dp.message(F.text == "🔐 Запросить доступ")
async def request_access(m: Message):
    await ensure_user(m.from_user.id)

    status, approved_until, requested_at = await get_user_access(m.from_user.id)

    if is_access_active(status, approved_until):
        await m.answer("✅ У тебя уже есть активный доступ. Нажми <b>📍 Новый сигнал</b>.", reply_markup=main_kb())
        return

    if status == "blocked":
        await m.answer("⛔️ Доступ заблокирован. Напиши администратору.")
        return

    ts = now_ts()
    if requested_at and (ts - requested_at) < 120:
        await m.answer("⏳ Заявка уже отправлена. Подожди немного — я отвечу.", reply_markup=locked_kb())
        return

    await set_user_pending(m.from_user.id)

    username = f"@{m.from_user.username}" if m.from_user.username else "без username"
    await m.answer("✅ Заявка отправлена. Как только я одобрю — доступ включится на 30 дней.", reply_markup=locked_kb())

    try:
        await bot.send_message(
            ADMIN_ID,
            "🔔 <b>Запрос доступа</b>\n\n"
            f"User ID: <code>{m.from_user.id}</code>\n"
            f"Username: {username}\n"
            f"Name: {m.from_user.full_name}",
            reply_markup=admin_approve_kb(m.from_user.id)
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
            "ℹ️ <b>Как пользоваться ботом</b>\n\n"
            "1) Нажми <b>📍 Новый сигнал</b>.\n"
            "2) Бот пришлёт сигнал: направление (BUY/SELL), вход (Entry), цели (TP/SL).\n"
            "3) После выдачи сигнала бот начинает автоматически следить за ценой.\n"
            "4) Когда цена достигнет TP или SL — бот отправит уведомление.\n"
            "5) Пока сигнал активен — новый сигнал не выдаётся.\n\n"
            "Пары: <b>EUR/USD</b> и <b>XAU/USD</b>.\n"
            "Таймфреймы: <b>5M / 15M / 30M</b> (бот выбирает лучший из доступных).\n\n"
            "⚠️ Сигналы не гарантия прибыли."
        )
    else:
        await m.answer(
            "ℹ️ <b>Помощь</b>\n\n"
            "Чтобы пользоваться сигналами, нужен доступ от администратора.\n"
            "Нажми <b>🔐 Запросить доступ</b>.\n"
            f"После одобрения доступ действует <b>{ACCESS_DAYS} дней</b>, затем нужно одобрение снова.",
            reply_markup=locked_kb()
        )


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
    await m.answer(signal_text(sig, note=best_note), reply_markup=main_kb())
    await start_watch(m.from_user.id)


@dp.callback_query(F.data.startswith("approve:"))
async def cb_approve(q: CallbackQuery):
    if q.from_user.id != ADMIN_ID:
        await q.answer("Недостаточно прав.", show_alert=True)
        return

    try:
        user_id = int(q.data.split("approve:")[1])
    except Exception:
        await q.answer("Ошибка данных.", show_alert=True)
        return

    await ensure_user(user_id)
    until = await approve_user_30d(user_id)

    until_dt = datetime.utcfromtimestamp(until).strftime("%Y-%m-%d %H:%M UTC")

    await q.message.edit_text(
        q.message.text + "\n\n✅ <b>Одобрено на 30 дней</b>\n" + f"Действует до: <code>{until_dt}</code>"
    )
    await q.answer("Одобрено.")

    try:
        await bot.send_message(
            user_id,
            "✅ Доступ активирован!\n\n"
            f"Теперь у тебя есть доступ на <b>{ACCESS_DAYS} дней</b>.\n"
            "Нажми <b>📍 Новый сигнал</b>.",
            reply_markup=main_kb()
        )
    except Exception:
        pass


@dp.callback_query(F.data.startswith("block:"))
async def cb_block(q: CallbackQuery):
    if q.from_user.id != ADMIN_ID:
        await q.answer("Недостаточно прав.", show_alert=True)
        return

    try:
        user_id = int(q.data.split("block:")[1])
    except Exception:
        await q.answer("Ошибка данных.", show_alert=True)
        return

    await ensure_user(user_id)
    await block_user(user_id)

    await q.message.edit_text(q.message.text + "\n\n⛔️ <b>Пользователь заблокирован</b>")
    await q.answer("Заблокировано.")

    try:
        await bot.send_message(
            user_id,
            "⛔️ Доступ заблокирован. Если это ошибка — напиши администратору."
        )
    except Exception:
        pass


async def main():
    await db_init()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
