
import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Dict, List

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    InputMediaPhoto,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv
import replicate

# ----------------------------
# Setup
# ----------------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
MODEL_INSTANTID = os.getenv("REPLICATE_MODEL_INSTANTID", "")
MODEL_PULID = os.getenv("REPLICATE_MODEL_PULID", "bytedance/flux-pulid:latest")

if not BOT_TOKEN:
    raise SystemExit("Please set BOT_TOKEN in .env")
if not REPLICATE_API_TOKEN:
    raise SystemExit("Please set REPLICATE_API_TOKEN in .env")

bot = Bot(BOT_TOKEN, parse_mode="HTML")
dp = Dispatcher(storage=MemoryStorage())
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# ----------------------------
# Styles
# ----------------------------
STYLES = Dict[str, str] = {
    "cinematic": "award-winning cinematic portrait, 85mm f/1.8 look, shallow depth of field, creamy bokeh, dramatic key light, rich filmic color grade, subtle grain, volumetric atmosphere, natural skin texture",
    "film35": "authentic 35mm film photograph, Portra-style color, fine grain, halation bloom, gentle contrast, true-to-life skin tones, slight vignette",
    "editorial_bw": "high-end editorial black & white portrait, soft directional light, deep tonality, silver-gelatin look, crisp micro-contrast, seamless studio backdrop, timeless aesthetic",
    "headshot_linkedin": "clean corporate headshot, flattering soft key + fill, neutral seamless background, accurate skin tone, light natural retouch, sharp eyes, friendly confident expression",
    "fashion": "glossy fashion editorial photo, beauty dish with rim light, studio set, precise makeup, high micro-detail, magazine cover polish, luxury styling",
    "beauty_soft": "beauty portrait close-up, large softbox wrap lighting, luminous complexion, delicate pore-level texture, airy pastel palette, subtle dodge & burn feel",
    "oil_impasto": "fine-art oil painting portrait, pronounced impasto, visible brushwork, layered pigments, chiaroscuro lighting, museum display mood",
    "watercolor_ink": "mixed-media watercolor & ink illustration, cold-press paper texture, controlled bleed, elegant linework, soft washes, hand-drawn feel",
    "pencil_drawing": "hyper-real graphite pencil drawing, fine cross-hatching, visible paper tooth, precise shading, classical life-drawing study",
    "pastel_dream": "soft pastel fine-art portrait, velvety chalk texture, gentle gradients, muted palette, romantic haze",
    "comic_halftone": "bold comic book panel, crisp ink lines, Ben-Day halftone shading, dynamic framing, screen-print texture",
    "poster_pop": "modern pop-graphic poster, clean vectors, geometric forms, striking color blocks, tight Swiss-grid composition",
    "sticker_cutout": "die-cut sticker look, white stroke border, flat shading, bold silhouette, playful kawaii vibe",
    "clay_3d": "studio 3D clay render, matte sculpted clay, single area light, soft subsurface scattering, seamless cyclorama",
    "toon_3d_clean": "clean toon-shaded 3D render, smooth surfaces, simple PBR materials, soft GI, rounded edges, tidy outlines",
    "vaporwave": "retro-futurist vaporwave scene, neon gradients, chrome typography, grid horizon, palm trees, nostalgic 80s ambiance",
    "y2k_gloss": "Y2K glossy tech aesthetic, pearlescent plastic, chrome accents, lens-flare gleam, iridescent highlights",
    "cyberpunk": "cinematic cyberpunk portrait, rainy neon street, holographic HUD accents, moody rim lights, reflective wet surfaces, intricate details",
    "steampunk": "ornate steampunk portrait, brass & leather fittings, cogs and valves, warm workshop glow, aged patina, Victorian flavor",
    "dark_fantasy": "epic dark fantasy portrait, drifting fog, dramatic rim light, gothic atmosphere, intricate armor, cinematic depth",
    "nature_fineart": "fine-art environmental portrait, overcast soft light, shallow DOF, natural color harmony, gentle breeze, calm mood",
    "desert_gold": "golden-hour desert portrait, warm backlight, dust haze, long shadows, sun flare, wide-open aperture glint",
    "isometric": "isometric game-style render, pristine edges, soft global illumination, miniature diorama feel, subtle contact shadows",
    "minimal_mag": "minimalist editorial layout, generous negative space, elegant typography, restrained palette, balanced composition"
}

# (опционально) общий негативный промпт
NEGATIVE = "low-res, harsh flash, overexposed, underexposed, plastic skin, oversharpened, artifacts, watermark, text, logo, extra fingers, deformed hands, cross-eye, duplicate, blur"


PACKS: Dict[str, List[str]] = {
    "starter10": [
        "cinematic", "cyberpunk", "fashion", "watercolor_ink", "oil_impasto",
        "vaporwave", "isometric", "steampunk", "clay_3d", "film35",
    ],
    "real6": [
        "cinematic", "film35", "headshot_linkedin", "editorial_bw", "beauty_soft", "nature_fineart"
    ],
    "art6": [
        "oil_impasto", "watercolor_ink", "pencil_drawing", "pastel_dream", "poster_pop", "comic_halftone"
    ],
    "future6": [
        "cyberpunk", "steampunk", "vaporwave", "y2k_gloss", "clay_3d", "toon_3d_clean"
    ],
}

# ----------------------------
# UI
# ----------------------------
def engine_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    if MODEL_INSTANTID:
        kb.button(text="InstantID / SDXL", callback_data="engine:instantid")
    kb.button(text="FLUX ID (PuLID)", callback_data="engine:pulid")
    kb.adjust(1)
    return kb.as_markup()

def style_kb(include_pack: bool = True) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for key in STYLES.keys():
        kb.button(text=key, callback_data=f"style:{key}")
    if include_pack:
        kb.button(text="🎁 10-pack (Starter)", callback_data="pack:starter10")
        kb.button(text="📦 Packs...", callback_data="packs:menu")
    kb.adjust(2)
    return kb.as_markup()

def packs_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for pack_id in PACKS.keys():
        kb.button(text=pack_id, callback_data=f"pack:{pack_id}")
    kb.button(text="⬅️ Back to styles", callback_data="packs:back")
    kb.adjust(2)
    return kb.as_markup()

# ----------------------------
# Replicate wrappers
# ----------------------------
def _replicate_run(model: str, inputs: dict, *, flatten: bool = True):
    out = replicate_client.run(model, input=inputs)
    if flatten and isinstance(out, list):
        return out[0]
    return out

def gen_pulid(
    image_url: str,
    style_prompt: str,
    seed: int = 0,
    w: int = 1024,
    h: int = 1024,
    num_outputs: int = 1,
    start_step: int = 0,
    id_weight: float = 1.0,
    guidance_scale: float = 4.0,
    num_steps: int = 20,
    true_cfg: float = 1.0,
    output_format: str = "png",
    output_quality: int = 90,
    max_sequence_length: int = 128,
    model_slug: str = MODEL_PULID,
) -> List[str]:
    prompt = f"{style_prompt}, detailed facial features, flux aesthetic, crisp details"
    negative = "low quality, distorted face, bad proportions, jpeg artifacts, blurry"
    inputs = {
        "main_face_image": image_url,
        "prompt": prompt,
        "negative_prompt": negative,
        "width": w,
        "height": h,
        "num_steps": num_steps,
        "start_step": start_step,
        "guidance_scale": guidance_scale,
        "id_weight": id_weight,
        "seed": seed,
        "true_cfg": true_cfg,
        "max_sequence_length": max_sequence_length,
        "output_format": output_format,
        "output_quality": output_quality,
        "num_outputs": num_outputs,
    }
    raw = _replicate_run(model_slug, inputs, flatten=(num_outputs == 1))
    if isinstance(raw, list):
        urls: List[str] = []
        for item in raw:
            try:
                urls.append(item.url())
            except Exception:
                urls.append(str(item))
        return urls
    try:
        return [raw.url()]  # type: ignore[attr-defined]
    except Exception:
        return [str(raw)]

def gen_instantid_v2(
    image_url: str,
    style_prompt: str,
    cc_scale: float = 0.6,
    model_slug: str = MODEL_INSTANTID,
) -> str:
    inputs = {
        "image": image_url,
        "prompt": style_prompt,
        "controlnet_conditioning_scale": cc_scale,
    }
    raw = _replicate_run(model_slug, inputs, flatten=False)
    try:
        return raw.url()  # type: ignore[attr-defined]
    except Exception:
        return str(raw)

# ----------------------------
# State
# ----------------------------
@dataclass
class UserSession:
    ref_image_url: str = ""
    engine: str = "instantid"  # or "pulid"
    width: int = 1024
    height: int = 1024

# ----------------------------
# Helpers
# ----------------------------
async def get_telegram_file_url(message: Message) -> str:
    photo = message.photo[-1]
    f = await bot.get_file(photo.file_id)
    return f"https://api.telegram.org/file/bot{BOT_TOKEN}/{f.file_path}"

async def generate_single(
    engine: str,
    image_url: str,
    style_key: str,
    seed: int = 0,
    w: int = 1024,
    h: int = 1024,
    variations: int = 1,
) -> List[str]:
    style_prompt = STYLES[style_key]
    if engine == "instantid":
        if not MODEL_INSTANTID:
            raise RuntimeError("InstantID engine not configured. Set REPLICATE_MODEL_INSTANTID.")
        url = await asyncio.to_thread(gen_instantid_v2, image_url, style_prompt)
        return [url]
    urls = await asyncio.to_thread(gen_pulid, image_url, style_prompt, seed, w, h, variations)
    return urls

async def generate_pack(
    engine: str,
    image_url: str,
    styles: List[str],
    seed_base: int = 0,
    w: int = 1024,
    h: int = 1024,
) -> List[str]:
    sem = asyncio.Semaphore(4)

    async def worker(idx: int, style_key: str) -> str:
        async with sem:
            urls = await generate_single(engine, image_url, style_key, seed_base + idx, w, h, variations=1)
            return urls[0]

    tasks = [worker(i, k) for i, k in enumerate(styles)]
    return await asyncio.gather(*tasks)

# ----------------------------
# Handlers
# ----------------------------
@dp.message(CommandStart())
async def on_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Загрузи 1 фото лица (фронтально, хороший свет) — дальше выберем движок и стиль.")

@dp.message(F.photo)
async def on_photo(message: Message, state: FSMContext):
    image_url = await get_telegram_file_url(message)
    session = UserSession(ref_image_url=image_url)
    await state.update_data(session=session.__dict__)

    text = (
        "Фото получено! Выбери движок:\n"
        "- InstantID / SDXL — быстрый, предсказуемый (по умолчанию).\n"
        "- FLUX ID (PuLID) — модный FLUX-look, иногда каприз к промптам."
    )
    await message.answer(text, reply_markup=engine_kb())

@dp.callback_query(F.data.startswith("engine:"))
async def on_engine(call: CallbackQuery, state: FSMContext):
    engine = call.data.split(":", 1)[1]
    data = await state.get_data()
    session = UserSession(**data.get("session", {}))
    session.engine = engine
    await state.update_data(session=session.__dict__)

    engine_name = "InstantID / SDXL" if engine == "instantid" else "FLUX ID (PuLID)"
    text = (f"Движок: <b>{engine_name}</b>\n"
            "Выбери стиль или нажми на пак:")
    await call.message.answer(text, reply_markup=style_kb(include_pack=True))
    await call.answer()

@dp.callback_query(F.data.startswith("style:"))
async def on_style(call: CallbackQuery, state: FSMContext):
    style_key = call.data.split(":", 1)[1]
    if style_key not in STYLES:
        await call.answer("Неизвестный стиль", show_alert=True)
        return

    data = await state.get_data()
    session = UserSession(**data.get("session", {}))
    if not session.ref_image_url:
        await call.answer("Сначала загрузите фото.", show_alert=True)
        return

    await call.message.answer(f"Генерирую: <b>{style_key}</b> …")
    try:
        urls = await generate_single(session.engine, session.ref_image_url, style_key)
    except Exception as e:
        await call.message.answer(f"Ошибка генерации: {e}")
        await call.answer()
        return

    if len(urls) == 1:
        await call.message.answer_photo(urls[0], caption=style_key)
    else:
        media = [InputMediaPhoto(media=u, caption=f"{style_key} #{i+1}") for i, u in enumerate(urls)]
        await call.message.answer_media_group(media)
    await call.answer()

@dp.callback_query(F.data == "packs:menu")
async def on_packs_menu(call: CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup(reply_markup=packs_kb())
    await call.answer()

@dp.callback_query(F.data == "packs:back")
async def on_packs_back(call: CallbackQuery, state: FSMContext):
    await call.message.edit_reply_markup(reply_markup=style_kb(include_pack=True))
    await call.answer()

@dp.callback_query(F.data.startswith("pack:"))
async def on_pack(call: CallbackQuery, state: FSMContext):
    pack_id = call.data.split(":", 1)[1]
    if pack_id not in PACKS:
        await call.answer("Неизвестный пак", show_alert=True)
        return

    data = await state.get_data()
    session = UserSession(**data.get("session", {}))
    if not session.ref_image_url:
        await call.answer("Сначала загрузите фото.", show_alert=True)
        return

    styles = PACKS[pack_id]
    await call.message.answer(f"Генерирую пак <b>{pack_id}</b> из {len(styles)} изображений…")

    try:
        urls = await generate_pack(session.engine, session.ref_image_url, styles)
    except Exception as e:
        await call.message.answer(f"Ошибка генерации пака: {e}")
        await call.answer()
        return

    media = [InputMediaPhoto(media=u, caption=styles[i]) for i, u in enumerate(urls)]
    await call.message.answer_media_group(media)
    await call.answer()

# ----------------------------
# Entry
# ----------------------------
async def main():
    print("Bot is running...", flush=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped.")
