import asyncio
import io
import os
import logging
import zipfile
import tempfile
from typing import Dict, Any, List, Optional

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

# Бэкенды и модели (можно переопределять через ENV)
INSTANTID_MODEL   = os.getenv("INSTANTID_MODEL", "grandlineai/instant-id-photorealistic")
INSTANTID_VERSION = os.getenv("INSTANTID_VERSION", "").strip()  # хэш версии (опц.)

IPADAPTER_MODEL   = os.getenv("IPADAPTER_MODEL", "lucataco/ip-adapter-faceid")
IPADAPTER_VERSION = os.getenv("IPADAPTER_VERSION", "").strip()

FLUX_MODEL        = os.getenv("FLUX_MODEL", "black-forest-labs/flux-1.1-pro-ultra")
FLUX_VERSION      = os.getenv("FLUX_VERSION", "").strip()

# finetuned-модель (инференс по finetune_id)
FINETUNED_MODEL   = os.getenv("FINETUNED_MODEL", "black-forest-labs/flux-1.1-pro-ultra-finetuned")
FINETUNED_VERSION = os.getenv("FINETUNED_VERSION", "").strip()

# тренер (официальный) — задай в ENV для обучения
LORA_TRAINER_MODEL = os.getenv("LORA_TRAINER_MODEL", "")   # напр. "black-forest-labs/flux-pro-trainer"

# для LoRA-URL (если будешь использовать не finetune_id, а веса)
LORA_APPLY_PARAM     = os.getenv("LORA_APPLY_PARAM", "lora_urls")  # или "adapters"
LORA_SCALE_DEFAULT   = float(os.getenv("LORA_SCALE_DEFAULT", "0.7"))

# ===================== SDK / BOT =====================
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

# ===================== IN-MEMORY STATE (на проде вынеси в Redis/DB) =====================
USER_LAST_PHOTO: Dict[int, bytes] = {}
USER_LAST_PROMPT: Dict[int, str] = {}
USER_BACKEND: Dict[int, str] = {}   # "instantid" | "ipadapter" | "flux"
USER_TRAIN_SET: Dict[int, List[bytes]] = {}
# Вариант finetune: {"finetune_id": str|None, "trigger": str|None, "enabled": bool, "status": "none|training|ready|failed"}
# Вариант LoRA-URL: {"url": str|None, "enabled": bool, "status": "..."}  (оставлено на будущее)
USER_LORA: Dict[int, Dict[str, Any]] = {}

# ===================== STYLES (обновлённый набор) =====================
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
    # Новые стили вместо noir / pixar3d / anime:
    "synthwave_neon": {"prompt": "synthwave neon portrait, 1980s retrofuturism, gradient sky, grid horizon, glowing rim light, high contrast, vibrant magenta and cyan, glossy finish"},
    "watercolor_paper": {"prompt": "hand-painted watercolor portrait on textured paper, soft edges, delicate washes, natural pigments, subtle bleed, organic imperfections, serene mood"},
    "mag_cutout": {"prompt": "magazine cutout collage portrait, torn paper edges, halftone dots, bold typography accents, layered shapes, playful composition, print texture"}
}

def styles_keyboard() -> InlineKeyboardMarkup:
    buttons, row = [], []
    for k in STYLES.keys():
        row.append(InlineKeyboardButton(text=k, callback_data=f"style:{k}"))
    # разбивка на строки по 3
    grid = [list(keys) for keys in zip(*[iter(STYLES.keys())]*3)]
    buttons = [[InlineKeyboardButton(text=k, callback_data=f"style:{k}") for k in row3] for row3 in grid]
    rest = list(STYLES.keys())[len(grid)*3:]
    if rest:
        buttons.append([InlineKeyboardButton(text=k, callback_data=f"style:{k}") for k in rest])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def modes_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🔒 InstantID (1 фото)", callback_data="mode:instantid"),
            InlineKeyboardButton(text="🔒 IP-Adapter", callback_data="mode:ipadapter"),
            InlineKeyboardButton(text="🎨 FLUX", callback_data="mode:flux"),
        ]
    ])

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

# ===================== GENERATORS =====================
def generate_with_instantid(face_bytes: bytes, prompt: str) -> bytes:
    face = io.BytesIO(face_bytes); face.name = "face.jpg"
    inputs = {
        "image": face,
        "prompt": prompt,
        "controlnet_conditioning_scale": 0.6,
    }
    ref = _ref_with_version(INSTANTID_MODEL, INSTANTID_VERSION)
    try:
        out = replicate.run(ref, input=inputs)
        return _to_bytes_from_output(out)
    except ReplicateError as e:
        # мягкий фолбэк на IP-Adapter, если порт недоступен/404
        if "404" in str(e) or "not be found" in str(e).lower():
            return generate_with_ipadapter(face_bytes, prompt)
        raise

def generate_with_ipadapter(face_bytes: bytes, prompt: str) -> bytes:
    face = io.BytesIO(face_bytes); face.name = "face.jpg"
    ref = _ref_with_version(IPADAPTER_MODEL, IPADAPTER_VERSION)
    try:
        out = replicate.run(ref, input={"face_image": face, "prompt": prompt})
    except Exception:
        face.seek(0)
        out = replicate.run(ref, input={"image": face, "prompt": prompt})
    return _to_bytes_from_output(out)

def generate_with_flux(image_bytes: bytes, prompt: str, lora_url: Optional[str] = None, lora_scale: float = LORA_SCALE_DEFAULT) -> bytes:
    image_file = io.BytesIO(image_bytes); image_file.name = "input.jpg"
    inputs: Dict[str, Any] = {
        "prompt": prompt,
        "image_prompt": image_file,
        "image_prompt_strength": 0.7,   # сильнее держим лицо/композицию
        "raw": True,
        "aspect_ratio": "1:1",
    }
    if lora_url:
        if LORA_APPLY_PARAM == "lora_urls":
            inputs["lora_urls"] = [lora_url]
            inputs["lora_scales"] = [lora_scale]
        elif LORA_APPLY_PARAM == "adapters":
            inputs["adapters"] = [{"path": lora_url, "weight": lora_scale}]
        else:
            inputs["lora_urls"] = [lora_url]

    ref = _ref_with_version(FLUX_MODEL, FLUX_VERSION)
    out = replicate.run(ref, input=inputs)
    return _to_bytes_from_output(out)

def generate_with_flux_finetuned(prompt: str, finetune_id: str, finetune_strength: float = 1.0) -> bytes:
    """Инференс по finetune_id (официальная finetuned-модель)."""
    ref = _ref_with_version(FINETUNED_MODEL, FINETUNED_VERSION)
    out = replicate.run(ref, input={
        "prompt": prompt,
        "finetune_id": finetune_id,
        "finetune_strength": finetune_strength,  # 0..2 (обычно 0.7–1.2)
        "raw": True,
        "aspect_ratio": "1:1",
    })
    return _to_bytes_from_output(out)

# ===================== FINETUNE TRAINING =====================
def _train_flux_finetune_sync(image_bytes_list: List[bytes], trigger_word: str) -> str:
    """Синхронный запуск тренера (репозиторий укажи в LORA_TRAINER_MODEL)."""
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        with zipfile.ZipFile(tmp, "w") as zf:
            for i, b in enumerate(image_bytes_list):
                zf.writestr(f"img_{i:02d}.jpg", b)
        tmp.flush()
        zip_path = tmp.name

    if not LORA_TRAINER_MODEL:
        raise RuntimeError("LORA_TRAINER_MODEL is not set")

    # офиц. тренер FLUX ожидает input_images (zip) + trigger_word + training_steps и т.п.
    ref = _ref_with_version(LORA_TRAINER_MODEL, "")
    out = replicate.run(ref, input={
        "input_images": open(zip_path, "rb"),
        "trigger_word": trigger_word,
        "training_steps": 300,
        "finetune_type": "lora",
        # "lora_rank": 32, "learning_rate": 1e-4, ...
    })
    if isinstance(out, str):
        return out
    if isinstance(out, dict) and "finetune_id" in out:
        return out["finetune_id"]
    raise RuntimeError(f"Unexpected trainer output: {type(out)}")

async def _train_and_notify(uid: int, imgs: List[bytes]):
    """Фоновые обучение finetune, чтобы не блокировать бота."""
    trigger = f"user_{uid}"  # рекомендуемо добавлять в промпт при инференсе
    loop = asyncio.get_running_loop()
    try:
        finetune_id = await loop.run_in_executor(None, lambda: _train_flux_finetune_sync(imgs, trigger))
        USER_LORA[uid] = {"finetune_id": finetune_id, "trigger": trigger, "enabled": True, "status": "ready"}
        await bot.send_message(uid, f"✅ Финетюн готов и включён!\nID: <code>{finetune_id}</code>\nРежим FLUX → /styles → стиль.")
    except Exception as e:
        USER_LORA[uid] = {"finetune_id": None, "trigger": None, "enabled": False, "status": "failed"}
        logging.exception("LoRA/finetune training failed")
        await bot.send_message(uid, f"⚠️ Ошибка тренировки: {e}")

# ===================== HANDLERS =====================
@dp.message(CommandStart())
async def start(m: Message):
    uid = m.from_user.id
    USER_BACKEND.setdefault(uid, "ipadapter")  # дефолт — стабильный IP-Adapter
    USER_LORA.setdefault(uid, {"finetune_id": None, "trigger": None, "enabled": False, "status": "none"})
    await m.answer(
        "👋 Привет! Это AI-аватар-бот.\n\n"
        "1) Пришли своё фото (лучше как <b>файл</b> — без сжатия).\n"
        "2) /styles — выбери стиль.\n"
        "3) /modes — режим: <b>InstantID</b> / <b>IP-Adapter</b> / <b>FLUX</b>.\n"
        "4) /train — собрать 6–12 фото (файлы) для финетюна → /finish.\n"
        "5) /lora_on, /lora_off — включить/выключить LoRA (если есть finetune).\n"
        "6) /set_finetune &lt;finetune_id&gt; — вручную указать finetune_id (если уже обучен).\n\n"
        f"Текущий режим: <b>{USER_BACKEND[uid]}</b>"
    )

@dp.message(Command("styles"))
async def list_styles(m: Message):
    await m.answer("Выбери стиль ниже 👇", reply_markup=styles_keyboard())

@dp.message(Command("modes"))
async def modes(m: Message):
    await m.answer("Выбери режим. Рекомендация: InstantID для быстрого сходства из 1 фото.", reply_markup=modes_keyboard())

@dp.callback_query(F.data.startswith("mode:"))
async def switch_mode(cq: CallbackQuery):
    uid = cq.from_user.id
    mode = cq.data.split(":", 1)[1]
    USER_BACKEND[uid] = mode
    await cq.message.answer(f"Режим переключён на: <b>{mode}</b>")
    await cq.answer()

@dp.message(F.content_type == ContentType.PHOTO)
async def on_photo(m: Message):
    b = await _download_as_bytes(m)
    USER_LAST_PHOTO[m.from_user.id] = b
    USER_LAST_PROMPT[m.from_user.id] = (m.caption or "").strip()
    await m.answer("📸 Фото получено. Нажми /styles и выбери стиль.")

@dp.message(F.content_type == ContentType.DOCUMENT, F.document.mime_type.in_({"image/jpeg","image/png"}))
async def on_image_doc(m: Message):
    b = await _download_as_bytes(m)  # без сжатия
    USER_LAST_PHOTO[m.from_user.id] = b
    USER_LAST_PROMPT[m.from_user.id] = (m.caption or "").strip()
    await m.answer("📎 Изображение получено без сжатия. Нажми /styles и выбери стиль.")

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

    info = USER_LORA.get(uid)
    backend = USER_BACKEND.get(uid, "ipadapter")

    await cq.message.answer(f"🎨 Генерирую: <b>{key}</b> … 10–40 сек.")
    await cq.answer()

    try:
        face_bytes = USER_LAST_PHOTO[uid]

        # Если выбран FLUX и готов finetune — используем его (TEXT→IMG)
        if backend == "flux" and info and info.get("enabled") and info.get("status") == "ready" and info.get("finetune_id"):
            trigger = info.get("trigger") or ""
            prompt_ft = (f"{trigger}, {prompt}").strip(", ")
            img_bytes = generate_with_flux_finetuned(prompt_ft, info["finetune_id"], finetune_strength=0.9)
        else:
            # иначе — face-lock (InstantID/IP-Adapter) или обычная стилизация FLUX
            if backend == "instantid":
                img_bytes = generate_with_instantid(face_bytes, prompt)
            elif backend == "ipadapter":
                img_bytes = generate_with_ipadapter(face_bytes, prompt)
            else:  # FLUX стилизация по фото
                lora_url = info.get("url") if info else None  # если будешь подавать LoRA-URL
                img_bytes = generate_with_flux(face_bytes, prompt, lora_url=lora_url)

        await bot.send_photo(
            chat_id=uid,
            photo=BufferedInputFile(img_bytes, filename=f"{key}.jpg"),
            caption=f"Готово! Режим: {backend}  |  Стиль: {key}"
        )
    except Exception as e:
        logging.exception("Generation error")
        await cq.message.answer(f"⚠️ Ошибка генерации: {e}\nПопробуй другой стиль/режим или другое фото.")

# ======== FINETUNE FLOW ========
@dp.message(Command("train"))
async def cmd_train(m: Message):
    uid = m.from_user.id
    USER_TRAIN_SET[uid] = []
    USER_LORA.setdefault(uid, {"finetune_id": None, "trigger": None, "enabled": False, "status": "none"})
    await m.answer(
        "Загрузи 6–12 фото как <b>файлы</b> (скрепка → файл). Когда закончишь, напиши /finish.\n"
        "Советы: разные ракурсы, свет, эмоции; лицо крупно; без тяжёлых фильтров."
    )

@dp.message(Command("finish"))
async def start_training(m: Message):
    uid = m.from_user.id
    imgs = USER_TRAIN_SET.get(uid, [])
    if len(imgs) < 6:
        return await m.answer("Нужно минимум 6 фото. Дозагрузи и снова /finish.")

    if not LORA_TRAINER_MODEL:
        return await m.answer(
            "⚠️ Тренер не настроен (переменная LORA_TRAINER_MODEL пустая).\n"
            "Задай её в Railway → Variables (напр.: black-forest-labs/flux-pro-trainer) и повтори."
        )

    USER_LORA[uid] = {"finetune_id": None, "trigger": None, "enabled": False, "status": "training"}
    await m.answer("🚀 Запускаю обучение финетюна… Сообщу, когда будет готов.")
    asyncio.create_task(_train_and_notify(uid, imgs))

@dp.message(Command("lora_on"))
async def lora_on(m: Message):
    uid = m.from_user.id
    info = USER_LORA.get(uid, {})
    if not (info.get("finetune_id") or info.get("url")):
        return await m.answer("LoRA/finetune ещё не задан. Сначала /train или /set_finetune &lt;finetune_id&gt;.")
    info["enabled"] = True
    info["status"] = info.get("status", "ready")
    USER_LORA[uid] = info
    await m.answer("LoRA/finetune включён ✅")

@dp.message(Command("lora_off"))
async def lora_off(m: Message):
    uid = m.from_user.id
    info = USER_LORA.get(uid, {})
    info["enabled"] = False
    USER_LORA[uid] = info
    await m.answer("LoRA/finetune выключен ⛔️")

@dp.message(Command("set_finetune"))
async def set_finetune(m: Message):
    uid = m.from_user.id
    parts = (m.text or "").split(maxsplit=1)
    if len(parts) < 2:
        return await m.answer("Использование: /set_finetune &lt;finetune_id&gt;")
    fid = parts[1].strip()
    USER_LORA[uid] = {"finetune_id": fid, "trigger": f"user_{uid}", "enabled": True, "status": "ready"}
    await m.answer("✅ finetune_id сохранён и включён. Режим FLUX → /styles.")

# Собираем изображения в датасет (как файлы)
@dp.message(F.content_type == ContentType.DOCUMENT, F.document.mime_type.in_({"image/jpeg","image/png"}))
async def collect_train_images(m: Message):
    uid = m.from_user.id
    if uid not in USER_TRAIN_SET:
        return  # игнор, если не /train
    b = await _download_as_bytes(m)
    USER_TRAIN_SET[uid].append(b)
    await m.answer(f"Добавлено фото в датасет. Сейчас: {len(USER_TRAIN_SET[uid])} шт.")

# Фолбэк (диагностика)
@dp.message()
async def fallback(m: Message):
    txt = (m.text or m.caption or "").strip()
    await m.answer("Я здесь 👋 Пришли фото (лучше как файл), затем /styles. Режимы: /modes  |  Справка: /start")
    logging.info(f"Fallback update: content_type={m.content_type!r} text={txt!r}")

# ===================== MAIN =====================
async def main():
    logging.info("Starting bot polling…")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
