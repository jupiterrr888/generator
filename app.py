"""
TG Avatar Bot: InstantID (SDXL) + FLUX-dev PuLID

- aiogram v3
- replicate python client
- Inline UI: engine selection, style selection, one-click packs

ENV:
  BOT_TOKEN=...
  REPLICATE_API_TOKEN=...
  # Model slugs (examples; set your own working versions)
  REPLICATE_MODEL_INSTANTID=grandlineai/instant-id-photorealistic:03914a0c3326bf44383d0cd84b06822618af879229ce5d1d53bef38d93b68279
  REPLICATE_MODEL_PULID=bytedance/flux-pulid:latest

Install:
  pip install aiogram~=3.6 replicate~=0.25 python-dotenv~=1.0
Run:
  python app.py
"""

import asyncio
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
# ENV & Clients
# ----------------------------
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
# Style Library (curated)
# ----------------------------
STYLES: Dict[str, str] = {
    # Realistic / Photo
    "cinematic": "dramatic cinematic portrait, film still, shallow depth of field, 85mm, bokeh, rich color grading",
    "film35": "35mm film photo look, subtle grain, natural skin tones, halation",
    "editorial_bw": "high-end editorial black and white, soft contrast, fine tonality, studio quality",
    "headshot_linkedin": "clean corporate headshot, soft key light, neutral background, natural retouch, professional",

    # Fashion / Beauty
    "fashion": "glossy fashion editorial, studio flash, beauty lighting, magazine cover aesthetic, high detail skin",
    "beauty_soft": "beauty portrait, softbox light, smooth gradients, delicate skin texture",

    # Painterly / Traditional Art
    "oil_impasto": "oil painting impasto, thick textured brush strokes, museum lighting, intricate details",
    "watercolor_ink": "watercolor and ink, paper texture, soft bleed, hand-drawn outlines",
    "pencil_drawing": "fine graphite pencil drawing, cross-hatching, paper texture, realistic",
    "pastel_dream": "soft pastel painting, velvety texture, gentle transitions, fine art look",

    # Graphic / Stylized
    "comic_halftone": "bold comic illustration, halftone shading, inking lines, print texture",
    "poster_pop": "bold poster design, geometric shapes, clean vectors, modern pop-graphic style",
    "sticker_cutout": "sticker style, white cutout border, flat shading, playful",

    # 3D / CG
    "clay_3d": "3d clay sculpt render, single key light, subtle subsurface scattering, matte clay material",
    "toon_3d_clean": "clean 3D toon render, simple materials, soft GI, smooth edges",

    # Retro / Aesthetic
    "vaporwave": "80s retrofuturism vaporwave, neon gradients, grid horizon, palm trees",
    "y2k_gloss": "Y2K glossy aesthetic, chrome accents, plastic sheen, iridescent",

    # Worlds / Moods
    "cyberpunk": "moody neon city at night, rain reflections, high contrast, dystopian vibes",
    "steampunk": "ornate brass and leather, valves, patina, vintage workshop ambiance",
    "dark_fantasy": "dark fantasy portrait, atmospheric fog, dramatic rim light, epic mood",
    "nature_fineart": "fine art natural setting, soft overcast light, shallow depth, gentle color palette",
    "desert_gold": "golden desert light, warm sunset tones, dust haze, cinematic framing",

    # Design / Product-ish
    "isometric": "isometric game art render, soft global illumination, clean edges, subtle shadows",
    "minimal_mag": "minimal editorial layout, lots of negative space, typography-driven composition",
}

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
# UI Helpers
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
        kb.button(text="🎁 10‑пак (Starter)", callback_data="pack:starter10")
        kb.button(text="📦 Пакеты…", callback_data="packs:menu")
    kb.adjust(2)
    return kb.as_markup()


def packs_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for pack_id in PACKS.keys():
        kb.button(text=pack_id, callback_data=f"pack:{pack_id}")
    kb.button(text="⬅️ Назад к стилям", callback_data="packs:back")
    kb.adjust(2)
    return kb.as_markup()

# ----------------------------
# Replicate wrappers
# ----------------------------

def _replicate_run(model: str, inputs: dict, *, flatten: bool = True):
    """Sync call; run in a thread when used from async handlers."""
    output = replicate_client.run(model, input=inputs)
    if flatten and isinstance(output, list):
        return output[0]
    return output


def build_common_prompts(style_prompt: str, engine: str):
    if engine == "instantid":
        positive = f"{style_prompt}, ultra-detailed face, natural skin, realistic eyes"
        negative = "low quality, deformed, extra fingers, artifacts, oversharpen, blurry"
        steps = 28
        guidance = 4.5
    else:  # pulid / flux-dev
        positive = f"{style_prompt}, detailed facial features, flux aesthetic, crisp details"
        negative = "low quality, distorted face, bad proportions, jpeg artifacts, blurry"
        steps = 28
        guidance = 3.5
    return positive, negative, steps, guidance


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
    """PuLID (FLUX-dev) call matching Replicate schema."""
    prompt, negative, _, _ = build_common_prompts(style_prompt, engine="pulid")
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
    """InstantID photorealistic minimal schema.
    Uses: image, prompt, controlnet_conditioning_scale
    Returns: direct URL string.
    """
    prompt = f"{style_prompt}"
    inputs = {
        "image": image_url,
        "prompt": prompt,
        "controlnet_conditioning_scale": cc_scale,
    }
    raw = _replicate_run(model_slug, inputs, flatten=False)
    try:
        return raw.url()  # type: ignore[attr-defined]
    except Exception:
        return str(raw)

# ----------------------------
# State / Models
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
    """Build a public file URL so Replicate can fetch it."""
    photo = message.photo[-1]  # largest
    f = await bot.get_file(photo.file_id)
    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{f.file_path}"
    return url


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
    else:
        urls = await asyncio.to_thread(
            gen_pulid, image_url, style_prompt, seed, w, h, variations
        )
        return urls


async def generate_pack(
    engine: str, image_url: str, styles: List[str], seed_base: int = 0, w: int = 1024, h: int = 1024
) -> List[str]:
    sem = asyncio.Semaphore(4)

    async def worker(idx: int, style_key: str) -> str:
        async with sem:
            urls = await generate_single(engine, image_url, style_key, seed_base + idx, w, h, variations=1)
            return urls[0]

    tasks = [worker(i, k) for i, k in enumerate(styles)]
    results = await asyncio.gather(*tasks)
    return results

# ----------------------------
# Handlers
# ----------------------------
@dp.message(CommandStart())
async def on_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        "Загрузи <b>1 фото лица</b> (фронтально, хороший свет) — дальше выберем движок и стиль."
    )


@dp.message(F.photo)
async def on_photo(message: Message, state: FSMContext):
    image_url = await get_telegram_file_url(message)
    session = UserSession(ref_image_url=image_url)
    await state.update_data(session=session.__dict__)

    await message.answer(
        "Фото получено! Выбери движок:
"
        "• <b>InstantID / SDXL</b> — быстрый, предсказуемый (по умолчанию).
"
        "• <b>FLUX ID (PuLID)</b> — модный FLUX-лук, иногда каприз к промптам.",
        reply_markup=engine_kb(),
    )


@dp.callback_query(F.data.startswith("engine:"))
async def on_engine(call: CallbackQuery, state: FSMContext):
    engine = call.data.split(":", 1)[1]
    data = await state.get_data()
    session = UserSession(**data.get("session", {}))
    session.engine = engine
    await state.update_data(session=session.__dict__)

    await call.message.answer(
        f"Движок: <b>{'InstantID / SDXL' if engine=='instantid' else 'FLUX ID (PuLID)'}</b>
"
        "Выбери стиль или нажми на пак:",
        reply_markup=style_kb(include_pack=True),
    )
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
        await call.message.answer_photo(urls[0], caption=f"{style_key}")
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
    print("Bot is running…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped.")
