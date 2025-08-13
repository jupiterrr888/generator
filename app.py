import os, asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ContentType
from aiogram.filters import CommandStart, Command
from aiogram.utils.markdown import hbold
from aiohttp import web
from dotenv import load_dotenv

from payments import create_checkout_session, stripe_webhook
from generator import generate_image

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
BASE_URL = os.getenv("BASE_URL")

bot = Bot(BOT_TOKEN, parse_mode="HTML")
dp = Dispatcher()

# Память на коленке (на проде - Redis/DB)
STATE = {}          # {user_id: {"style": str}}
BALANCES = {}       # {user_id: int}  # пополняется из вебхука
STYLES = {
    "anime": "high-quality anime portrait, soft lighting, detailed eyes",
    "cyberpunk": "cyberpunk portrait, neon lights, rain, bokeh, ultra-detailed",
    "realistic": "ultra realistic portrait, 85mm lens, soft studio light",
    "poster": "bold graphic poster, minimal layout, strong typography",
}

@dp.message(CommandStart())
async def start(m: Message):
    text = (
        f"Привет, {hbold(m.from_user.first_name)}!\n\n"
        "Это AI‑генератор фото.\n"
        "1) Выбери стиль: /styles\n"
        "2) Пополни баланс: /buy (10 генераций)\n"
        "3) Пришли фото и подпиши промпт‑идею.\n\n"
        f"Баланс: {BALANCES.get(m.from_user.id, 0)}"
    )
    await m.answer(text)

@dp.message(Command("styles"))
async def styles(m: Message):
    s = "\n".join([f"• {k} — {v}" for k,v in STYLES.items()])
    await m.answer("Доступные стили:\n" + s + "\n\nВыбери: /use_anime, /use_cyberpunk, /use_realistic, /use_poster")

@dp.message(F.text.startswith("/use_"))
async def use_style(m: Message):
    style_key = m.text.replace("/use_", "")
    if style_key not in STYLES:
        await m.answer("Такого стиля нет. Посмотри /styles")
        return
    STATE[m.from_user.id] = {"style": style_key}
    await m.answer(f"Стиль выбран: {style_key}. Теперь пришли фото и опиши идею (подпись к фото).")

@dp.message(Command("buy"))
async def buy(m: Message):
    url = create_checkout_session(m.from_user.id)
    await m.answer(f"Оплата за 10 генераций: {url}")

@dp.message(F.content_type == ContentType.PHOTO)
async def handle_photo(m: Message):
    user_id = m.from_user.id
    if BALANCES.get(user_id, 0) <= 0:
        await m.answer("Баланс 0. Пополнить: /buy")
        return
    st = STATE.get(user_id)
    if not st or "style" not in st:
        await m.answer("Сначала выбери стиль: /styles")
        return

    # Получаем ссылку на максимальное фото
    file_id = m.photo[-1].file_id
    f = await bot.get_file(file_id)
    image_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{f.file_path}"

    prompt_suffix = m.caption or ""  # подпись к фото как идея
    prompt = f"{STYLES[st['style']]} {prompt_suffix}".strip()

    await m.answer("Генерирую... это 10–30 секунд.")
    try:
        out_url = generate_image(prompt=prompt, image_url=image_url)
        # Списываем 1 генерацию
        BALANCES[user_id] = BALANCES.get(user_id, 0) - 1
        await m.answer_photo(out_url, caption=f"Готово!\nБаланс: {BALANCES.get(user_id, 0)}")
    except Exception as e:
        await m.answer(f"Ошибка генерации: {e}\nПроверь промпт или попробуй позже.")

# ---- Web server (вебхуки Stripe и Telegram) ----
async def on_startup(app: web.Application):
    app["balances"] = BALANCES

def make_app():
    app = web.Application()
    app.router.add_post("/stripe_webhook", stripe_webhook)
    return app

def main():
    app = make_app()
    loop = asyncio.get_event_loop()
    loop.create_task(dp.start_polling(bot))  # для простоты: long polling
    web.run_app(app, port=8080)

if __name__ == "__main__":
    main()
