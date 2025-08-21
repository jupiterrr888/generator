import asyncio
import io
import os
import logging
from typing import Dict, Any

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message, ContentType, InlineKeyboardMarkup, InlineKeyboardButton,
    CallbackQuery, BufferedInputFile
)
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from dotenv import load_dotenv

import requests
import replicate
from replicate.exceptions import ReplicateError

# ===================== ENV =====================
load_dotenv()
BOT_TOKEN = os.environ["BOT_TOKEN"]
REPLICATE_API_TOKEN = os.environ["REPLICATE_API_TOKEN"]

# InstantID (версия закрепляется через хэш)
INSTANTID_MODEL   = os.getenv("INSTANTID_MODEL", "grandlineai/instant-id-photorealistic")
INSTANTID_VERSION = os.getenv("INSTANTID_VERSION", "").strip()  # напр. 03914a0c33...

# ===================== SDK / BOT =====================
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

# ===================== IN-MEMORY STATE =====================
USER_LAST_PHOTO: Dict[int, bytes] = {}
USER_LAST_PROMPT: Dict[int, str] = {}

# ===================== STYLES =====================
STYLES: Dict[str, Dict[str, str]] = {
    "real_raw": {"prompt": "ultra realistic headshot, RAW photo look, soft studio light, skin texture, natural colors, detailed eyes, shallow depth of field, 85mm lens"},
    "cinematic": {"prompt": "cinematic portrait, dramatic rim light, volumetric light, film grain, teal and orange color grading, background bokeh"},
    "cyberpunk": {"prompt": "cyberpunk neon portrait, rainy night, neon reflections, holographic HUD, high detail, futuristic vibes"},
    "vogue": {"prompt": "editorial fashion portrait, glossy lighting, beauty retouch, professional studio, elegant styling, magazine cover look"},
    "lofi_film": {"prompt": "lofi film portrait, kodak portra vibes, soft highlights, muted tones, nostalgic mood, film grain subtle"},
    "fantasy_elf": {"prompt": "fantasy elf portrait, ethereal rim light, delicate face, intricate accessories, mystical forest atmosphere"},
    "comic": {"prompt": "comic book portrait, bold ink lines, halftone shading, high contrast, dynamic pose, graphic look"},
    "pop_art": {"prompt": "pop art portrait, bold colors, graphic shapes, clean background, posterized look, contemporary design"},
    "bw_classic": {"prompt": "black and white classic portrait, strong contrast, soft key light, timeless look, studio photography"},
    "poster_graphic": {"prompt": "graphic poster style portrait, minimal layout, strong typography accents, bold composition, color blocking"},
    "studio_beauty": {"prompt": "beauty studio portrait, clamshell lighting, glossy skin, catchlights in the eyes, editorial makeup"},
    "baroque": {"prompt": "baroque oil painting portrait, chiaroscuro, rich textures, ornate details, museum quality"},
    # новые:
    "synthwave_neon": {"prompt": "synthwave neon portrait, 1980s retrofuturism, gradient sky, grid horizon, glowing rim light, high contrast, vibrant magenta and cyan, glossy finish"},
    "watercolor_paper": {"prompt": "hand-painted watercolor portrait on textured paper, soft edges, delicate washes, natural pigments, subtle bleed, organic imperfections, serene mood"},
    "mag_cutout": {"prompt": "magazine cutout collage portrait, torn paper edges, halftone dots, bold typography accents, layered shapes, playful composition, print texture"},
}

def styles_keyboard() -> InlineKeyboardMarkup:
    buttons, row = [], []
    for k in STYLES.keys():
        row.append(InlineKeyboardButton(text=k, callback_data=f"style:{k}"))
        if len(row) == 3:
            buttons.append(row); row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(inline_keyboard=buttons)

# ===================== HELPERS =====================
def _to_bytes_from_output(output: Any) -> bytes:
    """Replicate может вернуть url, список, file-like. Приводим к bytes."""
    obj = output[0] if isinstance(output, list) else output
    if hasattr(obj, "read"):
        return obj.read()
    if isinstance(obj, str) and obj.startswith("http"):
        r = requests.get(obj, timeout=60)
        r.raise_for_status()
        return r.content
    raise RuntimeError("Unknown output type from Replicate model")

def _ref_with_version(model_ref: str, version: str) -> str:
    """Собираем ref с конкретной версией, если указана."""
    return f"{model_ref}:{version}" if version else model_ref

async def _download_as_bytes(message: Message) -> bytes:
    """Скачать изображение (photo/document) как bytes; для document Телега не сжимает."""
    buf = io.BytesIO()
    if message.content_type == ContentType.PHOTO:
        await bot.download(message.photo[-1], destination=buf)
    elif message.content_type == ContentType.DOCUMENT:
        await bot.download(message.document, destination=buf)
    else:
        raise ValueError("Unsupported content type for image download")
    buf.seek(0)
    return buf.getvalue()

# ===================== GENERATOR =====================
def generate_with_instantid(face_bytes: bytes, prompt: str) -> bytes:
    face = io.BytesIO(face_bytes); face.name = "face.jpg"
    inputs = {
        "image": face,                      # подаём файл (а не URL)
        "prompt": prompt,
        "controlnet_conditioning_scale": 0.6,
        # опционально: "num_inference_steps": 28, "guidance_scale": 3.5, "seed": 0,
    }
    ref = _ref_with_version(INSTANTID_MODEL, INSTANTID_VERSION)
    out = replicate.run(ref, input=inputs)
    return _to_bytes_from_output(out)

# ===================== HANDLERS =====================
@dp.message(CommandStart())
async def start(m: Message):
    await m.answer(
        "👋 Привет! Это AI-аватар-бот (InstantID только).\n\n"
        "1) Пришли своё фото (лучше как <b>файл</b> — без сжатия).\n"
        "2) Нажми /styles и выбери стиль — я пришлю аватар.\n\n"
        "Подсказка: крупное лицо анфас/3⁄4, без сильных фильтров."
    )

@dp.message(Command("styles"))
async def list_styles(m: Message):
    await m.answer("Выбери стиль ниже 👇", reply_markup=styles_keyboard())

@dp.message(F.content_type == ContentType.PHOTO)
async def on_photo(m: Message):
    b = await _download_as_bytes(m)   # photo (Телега сжимает, но ок)
    USER_LAST_PHOTO[m.from_user.id] = b
    USER_LAST_PROMPT[m.from_user.id] = (m.caption or "").strip()
    await m.answer("📸 Фото получено. Теперь нажми /styles и выбери стиль.")

@dp.message(F.content_type == ContentType.DOCUMENT, F.document.mime_type.in_({"image/jpeg","image/png"}))
async def on_image_doc(m: Message):
    b = await _download_as_bytes(m)   # без сжатия — лучше для сходства
    USER_LAST_PHOTO[m.from_user.id] = b
    USER_LAST_PROMPT[m.from_user.id] = (m.caption or "").strip()
    await m.answer("📎 Изображение получено без сжатия. Теперь нажми /styles и выбери стиль.")

@dp.callback_query(F.data.startswith("style:"))
async def on_style_click(cq: CallbackQuery):
    uid = cq.from_user.id
    key = cq.data.split(":", 1)[1]

    if uid not in USER_LAST_PHOTO:
        await cq.message.answer("Сначала пришли своё фото.")
        return await cq.answer()

    if key not in STYLES:
        await cq.message.answer("Такого стиля нет. Посмотри /styles")
        return await cq.answer()

    base_prompt = STYLES[key]["prompt"]
    extra = USER_LAST_PROMPT.get(uid, "")
    prompt = (base_prompt + (f", {extra}" if extra else "")).strip()

    await cq.message.answer(f"🎨 Генерирую: <b>{key}</b> … 10–40 сек.")
    await cq.answer()

    try:
        img_bytes = generate_with_instantid(USER_LAST_PHOTO[uid], prompt)
        await bot.send_photo(
            chat_id=uid,
            photo=BufferedInputFile(img_bytes, filename=f"{key}.jpg"),
            caption=f"Готово! Стиль: {key}"
        )
    except ReplicateError as e:
        logging.exception("InstantID Replicate error")
        await cq.message.answer(f"⚠️ Ошибка InstantID: {e}\nПроверь доступность версии модели или попробуй позже.")
    except Exception as e:
        logging.exception("Generation error")
        await cq.message.answer(f"⚠️ Ошибка генерации: {e}\nПопробуй другое фото или стиль.")

# Фолбэк (диагностика)
@dp.message()
async def fallback(m: Message):
    txt = (m.text or m.caption or "").strip()
    await m.answer("Я здесь 👋 Пришли фото (лучше как файл), затем /styles. Справка: /start")
    logging.info(f"Fallback update: content_type={m.content_type!r} text={txt!r}")

# ===================== MAIN =====================
async def main():
    logging.info("Starting bot polling…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
