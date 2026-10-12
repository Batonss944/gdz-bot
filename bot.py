"""
Телеграм-бот для вирішення домашки 🇺🇦
Працює на Python + aiogram 3 + безкоштовний ШІ
- Текстове завдання -> рішення з поясненням українською
- Фото завдання -> розпізнає і вирішує (Gemini Vision)
- Кнопки предметів для точніших відповідей
- Fallback: Gemini (якщо є ключ) -> Pollinations (без ключа, анонімно)

Запуск:
  pip install -r requirements.txt
  copy .env.example .env  (і вписати BOT_TOKEN)
  python bot.py
"""
import asyncio
import html
import io
import json
import logging
import os
import time
from datetime import datetime, timezone

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0") or 0)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Предмети (8 клас, 1 чверть, НУШ Україна) ---
SUBJECTS = [
    "📐 Алгебра", "📏 Геометрія",
    "⚛️ Фізика", "🧪 Хімія",
    "🌍 Географія", "🧬 Біологія",
    "🇺🇦 Історія України", "🏛️ Всесвітня історія",
    "📚 Укр. мова", "🇬🇧 Англійська",
    "📖 Інше",
]

def subjects_keyboard() -> ReplyKeyboardMarkup:
    kb = [
        [KeyboardButton(text=SUBJECTS[i]), KeyboardButton(text=SUBJECTS[i + 1])]
        for i in range(0, len(SUBJECTS) - 1, 2)
    ]
    if len(SUBJECTS) % 2 == 1:
        kb.append([KeyboardButton(text=SUBJECTS[-1])])
    return ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True)

# Запам'ятовуємо вибір предмета + антиспам (в пам'яті, для MVP достатньо)
user_subject: dict[int, str] = {}
user_last_req: dict[int, float] = {}

# --- Статистика користувачів ---
STATS_FILE = "stats.json"

def load_stats() -> dict:
    try:
        with open(STATS_FILE, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("users"), dict):
                return data
    except Exception:
        pass
    return {"users": {}}

def save_stats(data: dict):
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"stats save failed: {e}")

stats = load_stats()

def log_user(m: Message, kind: str, subject: str = "", preview: str = "") -> bool:
    """Записує користувача в stats.json. Повертає True, якщо користувач НОВИЙ."""
    u = m.from_user
    uid = str(u.id)
    users = stats.setdefault("users", {})
    is_new = uid not in users
    entry = users.get(uid, {})
    entry["name"] = u.full_name
    entry["username"] = f"@{u.username}" if u.username else "—"
    entry["requests"] = entry.get("requests", 0) + 1
    entry[f"{kind}s"] = entry.get(f"{kind}s", 0) + 1
    entry["last_seen"] = datetime.now(timezone.utc).strftime("%d.%m %H:%M")
    if subject:
        entry["last_subject"] = subject
    if preview:
        entry["last_task"] = preview[:80]
    users[uid] = entry
    save_stats(stats)
    return is_new

async def notify_admin(bot: Bot, m: Message):
    """Сповіщення власнику про нового користувача."""
    if ADMIN_ID and m.from_user.id != ADMIN_ID:
        u = m.from_user
        nick = f"@{u.username}" if u.username else "—"
        try:
            await bot.send_message(
                ADMIN_ID,
                f"👤 Новий користувач!\nІм'я: {u.full_name}\nНік: {nick}\nID: {u.id}"
            )
        except Exception as e:
            logger.warning(f"admin notify failed: {e}")

SYSTEM_PROMPT = (
    "Ти — добрий репетитор для учня 8 класу української школи (зараз 1-ша чверть). "
    "Відповідай УКРАЇНСЬКОЮ мовою, на рівні 8 класу: пояснюй просто, за програмою НУШ 8 класу, "
    "не забігай наперед у теми старших класів і не використовуй складні терміни без пояснення. "
    "ВАЖЛИВО: відповідай ПРОСТИМ ТЕКСТОМ для Telegram. "
    "ЗАБОРОНЕНО: Markdown (###, **, *, __, ```), LaTeX ($, \\frac, \\cdot, \\times), HTML-теги. "
    "Формули пиши звичайними символами на одному рядку: 2 + 2 × 2 = 6. "
    "Замість \\cdot і \\times пиши ×, замість \\div пиши ÷, замість \\frac{a}{b} пиши a/b. "
    "Списки — звичайними рядками з дефісом або цифрою, без зірочок. "
    "Формат відповіді:\n"
    "✅ Відповідь: (коротко, один рядок)\n"
    "📝 Розв'язання:\n"
    "Крок 1: ...\n"
    "Крок 2: ...\n"
    "💡 Пояснення простими словами: ...\n"
    "Якщо завдання — фото, спочатку уважно прочитай текст з фото. "
    "Якщо не видно або не зрозуміло — попроси надіслати чіткіше фото. "
    "Не просто давай відповідь, а вчи розуміти."
)

import re

def clean_for_telegram(text: str) -> str:
    """Прибирає Markdown/LaTeX, щоб у Telegram не було сміття типу ### ** $ \\cdot."""
    t = text.strip()
    # code fences ```...``` -> лишаємо всередині
    t = re.sub(r"```(\w*\n)?(.*?)```", r"\2", t, flags=re.DOTALL)
    # інлайн `код` -> код
    t = re.sub(r"`([^`\n]+)`", r"\1", t)
    # $$...$$, $...$, \(...\), \[...\] -> лишаємо всередині
    t = re.sub(r"\$\$(.*?)\$\$", r"\1", t, flags=re.DOTALL)
    t = re.sub(r"\$(.*?)\$", r"\1", t, flags=re.DOTALL)
    t = re.sub(r"\\\((.*?)\\\)", r"\1", t)
    t = re.sub(r"\\\[(.*?)\\\]", r"\1", t)
    # LaTeX з бекслешем
    t = t.replace("\\cdot", "×").replace("\\times", "×")
    t = t.replace("\\div", "÷").replace("\\pm", "±")
    t = t.replace("\\leq", "≤").replace("\\geq", "≥").replace("\\neq", "≠")
    t = t.replace("\\approx", "≈").replace("\\infty", "∞")
    t = re.sub(r"\\sqrt\{([^}]*)\}", r"√(\1)", t)
    t = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", t)
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("\\,", " ").replace("\\;", " ").replace("\\:", " ").replace("\\!", "")
    t = t.replace("\\\\", "")
    # голі слова cdot/times/div без бекслеша (коли \ з'їдено): "2 cdot 2" -> "2 × 2"
    t = re.sub(r"(?<=[\d\s])cdot(?=[\d\s])", "×", t)
    t = re.sub(r"(?<=[\d\s])times(?=[\d\s])", "×", t)
    t = re.sub(r"(?<=[\d\s])div(?=[\d\s])", "÷", t, flags=re.IGNORECASE)
    # залишки одиночних бекслешів перед буквами: \n \t не чіпати, інше -> прибрати бекслеш
    t = re.sub(r"\\([a-zA-Z]+)", r"\1", t)
    # картинки ![alt](url) -> alt
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)
    # посилання [текст](url) -> текст
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    # заголовки ### -> прибрати решітки
    t = re.sub(r"(?m)^\s*#{1,6}\s*", "", t)
    # жирний **текст** і __текст__ -> текст
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    # закреслений ~~текст~~ -> текст
    t = re.sub(r"~~([^~]+)~~", r"\1", t)
    # курсив *текст* (тільки парний, щоб не зламати 2*2) -> текст
    t = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", t)
    # маркери списку "* пункт" -> "• пункт"
    t = re.sub(r"(?m)^\s*\*\s+", "• ", t)
    # зайві порожні рядки
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def build_prompt(task_text: str, subject: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nПредмет: {subject}\nЗавдання: {task_text}"

# --- AI: Gemini (основний, безкоштовний, новий SDK) ---
GEMINI_FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-flash-latest"]

def _gemini_retryable(err: Exception) -> bool:
    """True — стоит повторить (перегрузка/таймаут). False — бессмысленно (нет модели, плохой ключ)."""
    s = f"{type(err).__name__}: {err}".lower()
    fatal_marks = ("404", "not_found", "not found", "invalid_argument", "invalid model", "api key", "permission", "unauthenticated")
    if any(m in s for m in fatal_marks):
        return False
    return True

async def ask_gemini(prompt: str, image_bytes: bytes | None = None) -> str:
    """Повертає текст відповіді або кидає Exception. При перевантаженні (503) пробує запасні моделі."""
    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        raise RuntimeError("no gemini key")
    client = genai.Client(api_key=GEMINI_API_KEY)

    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    last_err: Exception | None = None
    for model_name in models:
        t0 = time.time()
        for attempt in range(2):
            try:
                if image_bytes:
                    img_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                    resp = await client.aio.models.generate_content(
                        model=model_name,
                        contents=[img_part, prompt],
                    )
                else:
                    resp = await client.aio.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
                    )
                text = (resp.text or "").strip()
                if text:
                    logger.info(f"Gemini {model_name} OK in {time.time() - t0:.1f}s")
                    return text
                raise RuntimeError("empty response")
            except Exception as e:
                last_err = e
                logger.warning(f"Gemini {model_name} attempt {attempt + 1} failed in {time.time() - t0:.1f}s: {e}")
                if not _gemini_retryable(e):
                    break  # неверная модель/ключ — не ждём, сразу следующая модель
                await asyncio.sleep(1)
    raise last_err or RuntimeError("gemini failed")

# --- AI: Pollinations (запасний, БЕЗ ключа, анонімно) ---
async def ask_pollinations(prompt: str) -> str:
    """POST https://text.pollinations.ai/openai БЕЗ Authorization header."""
    url = "https://text.pollinations.ai/openai"
    payload = {
        "model": "openai",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 1500,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=90)) as r:
            if r.status != 200:
                text = await r.text()
                raise RuntimeError(f"pollinations {r.status}: {text[:200]}")
            data = await r.json()
            return data["choices"][0]["message"]["content"].strip()

async def ask_pollinations_vision(prompt: str, image_bytes: bytes) -> str:
    """Запасне розпізнавання фото через Pollinations (без ключа, анонімно). Кидає Exception при невдачі."""
    import base64

    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    url = "https://text.pollinations.ai/openai"
    payload = {
        "model": "openai",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.5,
        "max_tokens": 2000,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as r:
            if r.status != 200:
                text = await r.text()
                raise RuntimeError(f"pollinations vision {r.status}: {text[:200]}")
            data = await r.json()
            return data["choices"][0]["message"]["content"].strip()

async def solve_homework(task_text: str, subject: str, image_bytes: bytes | None = None) -> str:
    prompt = build_prompt(task_text, subject)
    t_start = time.time()
    # 1. Пробуємо Gemini (краще бачить фото, з ретраями)
    if GEMINI_API_KEY:
        try:
            ans = clean_for_telegram(await ask_gemini(prompt, image_bytes))
            logger.info(f"solve_homework via gemini in {time.time() - t_start:.1f}s")
            return ans
        except Exception as e:
            logger.warning(f"Gemini failed in {time.time() - t_start:.1f}s, fallback to pollinations: {e}")
    # 2. Фото: запасний варіант — Pollinations Vision
    if image_bytes:
        if not GEMINI_API_KEY:
            return (
                "📸 Я отримав фото, але для розпізнавання фото потрібен безкоштовний ключ Gemini.\n\n"
                "👉 Отримай за 1 хв: https://aistudio.google.com → Get API Key → встав у .env як GEMINI_API_KEY\n\n"
                "А поки — перепиши завдання ТЕКСТОМ, і я вирішу без ключа ✅"
            )
        try:
            ans = clean_for_telegram(await ask_pollinations_vision(prompt, image_bytes))
            logger.info(f"solve_homework via pollinations-vision in {time.time() - t_start:.1f}s")
            return ans
        except Exception as e:
            logger.warning(f"Pollinations vision failed in {time.time() - t_start:.1f}s: {e}")
            return (
                "😢 Сервери ШІ зараз перевантажені, фото не прочиталось.\n\n"
                "Спробуй так:\n"
                "1. Зачекай 1 хвилину і надішли фото ще раз\n"
                "2. Тримай телефон РІВНО, щоб текст був горизонтально, при хорошому світлі\n"
                "3. Або перепиши завдання текстом — так точно вийде ✅"
            )
    ans = clean_for_telegram(await ask_pollinations(prompt))
    logger.info(f"solve_homework via pollinations in {time.time() - t_start:.1f}s")
    return ans

def format_html(text: str) -> str:
    """Робить відповідь читабельною: жирна відповідь і заголовки блоків. Безпечно для Telegram HTML."""
    esc = html.escape(text, quote=False)
    esc = re.sub(r"(?m)^(✅\s*Відповідь\s*:.*)$", r"<b>\1</b>", esc)
    esc = re.sub(r"(?m)^(📝\s*Розв'язання\s*:?.*)$", r"<b>\1</b>", esc)
    esc = re.sub(r"(?m)^(💡.*)$", r"<b>\1</b>", esc)
    esc = re.sub(r"(?m)^(Крок\s*\d+\s*:)", r"<b>\1</b>", esc)
    return esc

async def send_long(m: Message, text: str, limit: int = 4000):
    """Відправляє довгий текст частинами по абзацах з HTML-форматуванням."""
    html_text = format_html(text)
    chunks: list[str] = []
    cur = ""
    for para in html_text.split("\n\n"):
        add = para if not cur else "\n\n" + para
        if len(cur) + len(add) <= limit:
            cur += add
        else:
            if cur:
                chunks.append(cur)
            while len(para) > limit:
                chunks.append(para[:limit])
                para = para[limit:]
            cur = para
    if cur:
        chunks.append(cur)
    for ch in chunks or ["..."]:
        try:
            await m.answer(ch, parse_mode="HTML")
        except Exception:
            plain = html.unescape(re.sub(r"</?b>", "", ch))
            await m.answer(plain)

def check_spam(user_id: int) -> bool:
    """True = можна, False = занадто часто."""
    now = time.time()
    last = user_last_req.get(user_id, 0)
    if now - last < 4:
        return False
    user_last_req[user_id] = now
    return True

# --- Handlers ---
dp = Dispatcher()

@dp.message(CommandStart())
async def cmd_start(m: Message, bot: Bot):
    user_subject[m.from_user.id] = "📖 Інше"
    if log_user(m, "start"):
        await notify_admin(bot, m)
    await m.answer(
        "Привіт! Я бот-помічник з домашки 🇺🇦\n\n"
        "📝 Надішли завдання <b>текстом</b>\n"
        "📸 Або надішли <b>фото</b> зошита / підручника\n\n"
        "Обери предмет кнопкою нижче 👇 — так я вирішуватиму точніше.",
        reply_markup=subjects_keyboard(),
        parse_mode="HTML",
    )

@dp.message(Command("help"))
async def cmd_help(m: Message):
    await m.answer(
        "/start — почати\n"
        "/predmet — змінити предмет\n"
        "Просто надішли текст або фото завдання 📚",
        reply_markup=subjects_keyboard(),
    )

@dp.message(Command("stats"))
async def cmd_stats(m: Message):
    if not ADMIN_ID or m.from_user.id != ADMIN_ID:
        await m.answer("⛔ Ця команда тільки для власника бота.")
        return
    users = stats.get("users", {})
    total_req = sum(u.get("requests", 0) for u in users.values())
    lines = [f"📊 Статистика: {len(users)} користувачів, {total_req} запитів"]
    top = sorted(users.items(), key=lambda kv: kv[1].get("requests", 0), reverse=True)[:15]
    for uid, u in top:
        lines.append(
            f"• {u.get('name')} ({u.get('username')}) — {u.get('requests')} зап., "
            f"фото {u.get('photos', 0)}/текст {u.get('texts', 0)}, ост. {u.get('last_seen', '?')}"
        )
    await send_long(m, "\n".join(lines))

@dp.message(Command("predmet"))
async def cmd_subject(m: Message):
    await m.answer("Обери предмет 👇", reply_markup=subjects_keyboard())

@dp.message(F.photo)
async def handle_photo(m: Message, bot: Bot):
    if not check_spam(m.from_user.id):
        await m.answer("⏳ Зачекай пару секунд і спробуй ще раз 🙏")
        return
    subject = user_subject.get(m.from_user.id, "📖 Інше")
    status = await m.answer("📸 Бачу фото, розпізнаю і вирішую... ⏳")

    try:
        photo = m.photo[-1]  # найбільший розмір
        buf = io.BytesIO()
        await bot.download(photo, destination=buf)
        image_bytes = buf.getvalue()

        caption = m.caption or "Розв'яжи завдання на фото."
        if log_user(m, "photo", subject, caption):
            await notify_admin(bot, m)
        answer = await solve_homework(caption, subject, image_bytes)
        await send_long(m, answer)
    except Exception as e:
        logger.exception("photo error")
        await m.answer(f"😢 Не вийшло обробити фото: {e}\nСпробуй чіткіше фото або напиши текстом.")
    finally:
        try:
            await status.delete()
        except Exception:
            pass

@dp.message(F.voice | F.audio)
async def handle_voice(m: Message):
    await m.answer(
        "🎤 Голосові поки не вмію розпізнавати в MVP.\n"
        "Напиши завдання текстом або надішли фото 📸 — вирішу одразу!"
    )

@dp.message(F.text)
async def handle_text(m: Message, bot: Bot):
    # Вибір предмета кнопкою
    if m.text in SUBJECTS:
        user_subject[m.from_user.id] = m.text
        await m.answer(f"Ок! Предмет: <b>{m.text}</b>\nТепер надсилай завдання 👇", parse_mode="HTML")
        return

    if not check_spam(m.from_user.id):
        await m.answer("⏳ Зачекай пару секунд і спробуй ще раз 🙏")
        return

    if len(m.text.strip()) < 3:
        await m.answer("Напиши завдання трохи детальніше 📝")
        return

    subject = user_subject.get(m.from_user.id, "📖 Інше")
    status = await m.answer("🧠 Думаю над рішенням... ⏳")
    try:
        if log_user(m, "text", subject, m.text):
            await notify_admin(bot, m)
        answer = await solve_homework(m.text, subject)
        await send_long(m, answer)
    except Exception as e:
        logger.exception("text error")
        await m.answer(
            "😢 Сталася помилка. Можливо, безкоштовний ліміт вичерпано (1 запит / 15 сек).\n"
            f"Зачекай 15 сек і спробуй ще раз. ({e})"
        )
    finally:
        try:
            await status.delete()
        except Exception:
            pass

async def health_server():
    """Міні-вебсервер для безкоштовного хостингу (Render Web Service + UptimeRobot пінгують /health і не дають заснути)."""
    from aiohttp import web

    async def ok(request):
        return web.Response(text="OK", status=200)

    app = web.Application()
    # GET + HEAD + все методы: Render шлёт HEAD, UptimeRobot — GET. Без HEAD будет 405 и Render снимет инстанс с ротации (no-server).
    for path in ("/", "/health"):
        app.router.add_route("*", path, ok)
    port = int(os.getenv("PORT", "10000"))
    runner = web.AppRunner(app, access_log=logging.getLogger("aiohttp.access"))
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"health server on port {port}")
    await asyncio.Event().wait()

async def main():
    if not BOT_TOKEN or "ABC-DEF" in BOT_TOKEN:
        print("[ERROR] Vstav sviy BOT_TOKEN u file .env (divis .env.example)")
        print("   Yak otrimati: Telegram -> @BotFather -> /newbot")
        return
    bot = Bot(token=BOT_TOKEN)
    has_gemini = "yes" if GEMINI_API_KEY else "no (text only via Pollinations)"
    print(f"[OK] Bot zapushcheno! Gemini: {has_gemini}")
    # поллінг Telegram + health-сервер одночасно (для Render Free).
    # health крутиться фоном, чтобы падение polling не роняло HTTP (иначе Render даст no-server).
    health_task = asyncio.create_task(health_server())
    try:
        await dp.start_polling(bot, drop_pending_updates=False)
    finally:
        health_task.cancel()
        try:
            await bot.session.close()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
