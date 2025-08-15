import asyncio
import io
import os
import tempfile
from contextlib import contextmanager

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, ContentType, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, BufferedInputFile
from dotenv import load_dotenv
import replicate
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
import logging


load_dotenv()
BOT_TOKEN = os.environ["BOT_TOKEN"]
REPLICATE_API_TOKEN = os.environ["REPLICATE_API_TOKEN"]
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML)
)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)



# Память «на коленке»: на проде вынести в Redis/БД
USER_LAST_PHOTO = {}        # user_id -> bytes последнего селфи
USER_LAST_PROMPT = {}       # user_id -> краткая идея/подпись

# Набор "крутых" стилей под аватарки (prompt пресеты)
STYLES = {
    "real_raw":        "ultra realistic headshot, RAW photo look, soft studio light, skin texture, natural colors, detailed eyes, shallow depth of field, 85mm lens",
    "cinematic":       "cinematic portrait, dramatic rim light, volumetric light, film grain, color grading teal and orange, bokeh background",
    "cyberpunk":       "cyberpunk neon portrait, rainy night, neon reflections, holographic UI elements, high detail, futuristic vibes",
    "anime":           "high-quality anime portrait, detailed eyes, clean lineart, soft shading, vibrant palette",
    "pixar3d":         "3d stylized portrait, pixar-like, subsurface scattering, soft key light, studio render, high detail",
    "vogue":           "editorial fashion portrait, glossy lighting, beauty retouch, professional studio, elegant styling",
    "lofi_film":       "lofi film portrait, kodak portra feel, soft highlights, muted tones, nostalgic mood",
    "fantasy_elf":     "fantasy elf portrait, delicate face, ethereal light, intricate accessories, mystical atmosphere",
    "comic":           "comic book portrait, bold ink lines, halftone shading, high contrast, dynamic look",
    "pop_art":         "pop art portrait, bold colors, graphic shapes, clean background, poster look",
    "bw_classic":      "black and white classic portrait, strong contrast, soft key light, timeless look",
    "poster_graphic":  "graphic poster style portrait, minimal layout, strong typography accents, bold composition"
}

def styles_keyboard():
    buttons = []
    row = []
    for k in STYLES.keys():
        row.append(InlineKeyboardButton(text=k, callback_data=f"style:{k}"))
        if len(row) == 3:
            buttons.append(row); row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(inline_keyboard=buttons)

@dp.message(CommandStart())
async def start(m: Message):
    await m.answer(
        "👋 Привет! Загрузите <b>своё фото/селфи</b> (как обычное фото).\n"
        "Потом просто выберите стиль на кнопке — и я пришлю готовую аватарку.\n\n"
        "Доступные стили: /styles"
    )

@dp.message(Command("styles"))
async def list_styles(m: Message):
    await m.answer("Выберите стиль ниже 👇", reply_markup=styles_keyboard())


@dp.message(F.content_type == ContentType.PHOTO)
async def on_photo(m: Message):
    # Скачиваем исходник в память (совместимо с aiogram 3.x)
    buf = io.BytesIO()
    try:
        # предпочтительный способ (есть в aiogram 3.x)
        await bot.download(m.photo[-1], destination=buf)
    except Exception:
        # fallback, если конкретно у тебя нет .download(...)
        file = await bot.get_file(m.photo[-1].file_id)
        await bot.download_file(file.file_path, buf)

    buf.seek(0)
    b = buf.getvalue()  # без await!

    USER_LAST_PHOTO[m.from_user.id] = b
    USER_LAST_PROMPT[m.from_user.id] = (m.caption or "").strip()

    await m.answer(
        "📸 Фото получено!\n"
        "Теперь выберите стиль — я сгенерирую аватарку.",
        reply_markup=styles_keyboard()
    )


@dp.callback_query(F.data.startswith("style:"))
async def on_style_click(cq: CallbackQuery):
    user_id = cq.from_user.id
    key = cq.data.split(":", 1)[1]

    if user_id not in USER_LAST_PHOTO:
        await cq.message.answer("Сначала пришлите своё фото (селфи).")
        await cq.answer()
        return

    if key not in STYLES:
        await cq.message.answer("Такого стиля нет. Посмотрите /styles")
        await cq.answer()
        return

    await cq.message.answer(f"🎨 Генерирую стиль: <b>{key}</b> … 10–30 сек.")
    await cq.answer()

    base_prompt = STYLES[key]
    extra = USER_LAST_PROMPT.get(user_id, "")
    prompt = (base_prompt + (f", {extra}" if extra else "")).strip()

    try:
        # Подготовим входной файл как file-like object (без внешних URL)
        image_bytes = USER_LAST_PHOTO[user_id]
        image_file = io.BytesIO(image_bytes)
        image_file.name = "input.jpg"  # имя нужно клиенту для multipart

        # Включим более «натуральный» вид и мягкое смешивание с исходником
        inputs = {
            "prompt": prompt,
            "image_prompt": image_file,            # локальный файл → безопасно
            "image_prompt_strength": 0.07,         # баланс стиль/композиция
            "raw": True,                           # более «реалистичный» тон
            "aspect_ratio": "1:1",                 # аватар-формат
            # можно поиграть c safety_tolerance (1..6) при необходимости
        }

        # Вызов модели (последняя версия по умолчанию)
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro-ultra",
            input=inputs
        )

        # Библиотека возвращает FileOutput; отправляем как photo из памяти
        if isinstance(output, list):
            file_output = output[0]
        else:
            file_output = output  # на некоторых моделях может быть одиночный

        img_bytes = file_output.read()
        await bot.send_photo(
            chat_id=user_id,
            photo=BufferedInputFile(img_bytes, filename=f"{key}.jpg"),
            caption=f"Готово! Стиль: {key}"
        )

    except Exception as e:
        await cq.message.answer(f"⚠️ Ошибка генерации: {e}\nПопробуйте другой стиль или другое фото.")
        
async def main():
    logging.info("Starting bot polling on Railway…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

