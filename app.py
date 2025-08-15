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

# ===================== ENV =====================
load_dotenv()
BOT_TOKEN = os.environ["BOT_TOKEN"]
REPLICATE_API_TOKEN = os.environ["REPLICATE_API_TOKEN"]

# Бэкенды и модели: можно переопределить через ENV в Railway
FLUX_MODEL = os.getenv("FLUX_MODEL", "black-forest-labs/flux-1.1-pro-ultra")
# Для финетюнов (только если используешь тренер, дающий finetune_id)
FINETUNED_MODEL = os.getenv("FINETUNED_MODEL", "black-forest-labs/flux-1.1-pro-ultra-finetuned")

# Face-lock модели (можно менять на свои)
INSTANTID_MODEL = os.getenv("INSTANTID_MODEL", "grandlineai/instant-id-photorealistic")
IPADAPTER_MODEL = os.getenv("IPADAPTER_MODEL", "lucataco/ip-adapter-faceid")

# Тренер LoRA/finetune на Replicate (ЗАДАЙ САМ!)
# Оставь пустым, если тренер ещё не выбран — /finish аккуратно сообщит об этом
LORA_TRAINER_MODEL = os.getenv("LORA_TRAINER_MODEL", "")  # пример: "black-forest-labs/flux-pro-trainer"

# Как подмешивается LoRA в обычный FLUX (если тренер возвращает URL весов)
LORA_APPLY_PARAM = os.getenv("LORA_APPLY_PARAM", "lora_urls")  # или "adapters" в зависимости от модели
LORA_SCALE_DEFAULT = float(os.getenv("LORA_SCALE_DEFAULT", "0.7"))

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
# Вариант 1 (без finetune_id): {"url": str|None, "enabled": bool, "status": "none|training|ready|failed"}
# Вариант 2 (с finetune_id):   {"finetune_id": str|None, "trigger": str|None, "enabled": bool, "status": "..."}
USER_LORA: Dict[int, Dict[str, Any]] = {}

# ===================== STYLES (расширенный набор) =====================
STYLES: Dict[str, Dict[str, str]] = {
    "real_raw": {"prompt": "ultra realistic headshot, RAW photo look, soft studio light, skin texture, natural colors, detailed eyes, shallow depth of field, 85mm lens"},
    "cinematic": {"prompt": "cinematic portrait, dramatic rim light, volumetric light, film grain, teal and orange color grading, background bokeh"},
    "cyberpunk": {"prompt": "cyberpunk neon portrait, rainy night, neon reflections, holographic HUD, high detail, futuristic vibes"},
    "anime": {"prompt": "masterpiece, best quality, high-quality anime portrait, detailed irises, clean lineart, soft shading, vibrant palette"},
    "pixar3d": {"prompt": "3d stylized pixar-like portrait, subsurface scattering, soft key light, studio render, highly detailed, glossy skin"},
    "vogue": {"prompt": "editorial fashion portrait, glossy lighting, beauty retouch, professional studio, elegant styling, magazine cover look"},
    "lofi_film": {"prompt": "lofi film portrait, kodak portra vibes, soft highlights, muted tones, nostalgic mood, film grain subtle"},
    "fantasy_elf": {"prompt": "fantasy elf portrait, ethereal rim light, delicate face, intricate accessories, mystical forest atmosphere"},
    "comic": {"prompt": "comic book portrait, bold ink lines, halftone shading, high contrast, dynamic pose, graphic look"},
    "pop_art": {"prompt": "pop art portrait, bold colors, graphic shapes, clean background, posterized look, contemporary design"},
    "bw_classic": {"prompt": "black and white classic portrait, strong contrast, soft key light, timeless look, studio photography"},
    "poster_graphic": {"prompt": "graphic poster style portrait, minimal layout, strong typography accents, bold composition, color blocking"},
    "studio_beauty": {"prompt": "beauty studio portrait, clamshell lighting, glossy skin, catchlights in the eyes, editorial makeup"},
    "noir": {"prompt": "film noir portrait, hard light, dramatic shadows, moody atmosphere, vintage cinematic look"},
    "baroque": {"prompt": "baroque oil painting portrait, chiaroscuro, rich textures, ornate details, museum quality"}
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
def generate_with_flux(image_bytes: bytes, prompt: str, lora_url: Optional[str] = None, lora_scale: float = LORA_SCALE_DEFAULT) -> bytes:
    image_file = io.BytesIO(image_bytes); image_file.name = "input.jpg"
    inputs: Dict[str, Any] = {
        "prompt": prompt,
        "image_prompt": image_file,
        "image_prompt_strength": 0.7,   # усилили влияние референса (для сходства)
        "raw": True,
        "aspect_ratio": "1:1",
    }
    # Если есть LoRA-URL (модели, поддерживающие подмешивание весов)
    if lora_url:
        if LORA_APPLY_PARAM == "lora_urls":
            inputs["lora_urls"] = [lora_url]
            inputs["lora_scales"] = [lora_scale]  # если модель поддерживает
        elif LORA_APPLY_PARAM == "adapters":
            inputs["adapters"] = [{"path": lora_url, "weight": lora_scale}]
        else:
            inputs["lora_urls"] = [lora_url]

    out = replicate.run(FLUX_MODEL, input=inputs)
    return _to_bytes_from_output(out)

def generate_with_flux_finetuned(prompt: str, finetune_id: str, finetune_strength: float = 1.0) -> bytes:
    """Инференс по finetune_id (если используешь модель вида *-finetuned)."""
    out = replicate.run(
        FINETUNED_MODEL,
        input={
            "prompt": prompt,
            "finetune_id": finetune_id,           # обязателен
            "finetune_strength": finetune_strength,  # 0..2 (варьируй 0.7–1.2)
            "raw": True,
            "aspect_ratio": "1:1",
        }
    )
    return _to_bytes_from_output(out)

def generate_with_instantid(face_bytes: bytes, prompt: str) -> bytes:
    face = io.BytesIO(face_bytes); face.name = "face.jpg"

    inputs = {
        "image": face,                      # вместо URL подаём файл
        "prompt": prompt,
        "controlnet_conditioning_scale": 0.6,   # как в твоём примере
        # при желании: "num_inference_steps": 28, "guidance_scale": 3.5, "seed": 0
    }

    # если у тебя есть _replicate_run(ref, inputs, version), используй его:
    out = _replicate_run(INSTANTID_MODEL, inputs, version=INSTANTID_VERSION)

    # файл/URL → bytes (у нас уже есть хелпер)
    return _to_bytes_from_output(out)


def generate_with_ipadapter(face_bytes: bytes, prompt: str) -> bytes:
    face = io.BytesIO(face_bytes); face.name = "face.jpg"
    # У разных портов поля могут отличаться — пробуем 2 варианта
    try:
        out = replicate.run(IPADAPTER_MODEL, input={"face_image": face, "prompt": prompt})
    except Exception:
        out = replicate.run(IPADAPTER_MODEL, input={"image": face, "prompt": prompt})
    return _to_bytes_from_output(out)

# ===================== LORA / FINETUNE TRAINING =====================
def _train_flux_finetune_sync(image_bytes_list: List[bytes], caption_prefix: str) -> str:
    """
    Синхронный запуск тренера на Replicate.
    ВАЖНО: поля inputs зависят от конкретной модели-тренера. Ниже пробуем несколько распространённых.
    Открой страницу твоего тренера на Replicate и при необходимости подгони имена полей.
    """
    # Собираем ZIP с картинками
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        with zipfile.ZipFile(tmp, "w") as zf:
            for i, b in enumerate(image_bytes_list):
                zf.writestr(f"img_{i:02d}.jpg", b)
        tmp.flush()
        zip_path = tmp.name

    if not LORA_TRAINER_MODEL:
        raise RuntimeError("LORA_TRAINER_MODEL is not set. Configure a trainer model on Replicate first.")

    # Пытаемся вызвать тренер с разными ключами датасета
    trainer_inputs_variants = [
        {"input_images": open(zip_path, "rb"), "caption_prefix": caption_prefix},
        {"images_zip": open(zip_path, "rb"), "caption_prefix": caption_prefix},
        {"dataset": open(zip_path, "rb"), "caption_prefix": caption_prefix},
        {"images": open(zip_path, "rb"), "caption_prefix": caption_prefix},
    ]

    last_exc = None
    for inp in trainer_inputs_variants:
        try:
            out = replicate.run(LORA_TRAINER_MODEL, input=inp)
            # Ожидаем либо finetune_id (str), либо dict c ключом
            if isinstance(out, str):
                return out
            if isinstance(out, dict):
                # На некоторых тренерах ключ может называться по-разному
                for k in ("finetune_id", "id", "model_id"):
                    if k in out and isinstance(out[k], str):
                        return out[k]
            if isinstance(out, list) and out and isinstance(out[0], str):
                return out[0]
        except Exception as e:
            last_exc = e
            continue

    raise RuntimeError(f"Trainer call failed / unexpected output. Last error: {last_exc}")

async def _train_and_notify(uid: int, imgs: List[bytes]):
    """Фоновые обучение finetune, чтобы не блокировать бота."""
    trigger = f"user_{uid}"  # будет рекомендовано добавлять в prompt
    loop = asyncio.get_running_loop()
    try:
        finetune_id = await loop.run_in_executor(None, lambda: _train_flux_finetune_sync(imgs, trigger))
        USER_LORA[uid] = {"finetune_id": finetune_id, "trigger": trigger, "enabled": True, "status": "ready"}
        await bot.send_message(uid, f"✅ Финетюн готов и включён!\nID: <code>{finetune_id}</code>\nРежим: FLUX → /styles → стиль.")
    except Exception as e:
        USER_LORA[uid] = {"finetune_id": None, "trigger": None, "enabled": False, "status": "failed"}
        logging.exception("LoRA/finetune training failed")
        await bot.send_message(uid, f"⚠️ Ошибка тренировки: {e}\nПроверь тренер (LORA_TRAINER_MODEL) и формат входов.")

# ===================== HANDLERS =====================
@dp.message(CommandStart())
async def start(m: Message):
    uid = m.from_user.id
    USER_BACKEND.setdefault(uid, "instantid")
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
    await m.answer("Выбери режим. Рекомендация: InstantID для сходства из 1 фото.", reply_markup=modes_keyboard())

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
    backend = USER_BACKEND.get(uid, "instantid")

    await cq.message.answer(f"🎨 Генерирую: <b>{key}</b> … 10–40 сек.")
    await cq.answer()

    try:
        # Если выбран FLUX и есть готовый finetune_id — используем его (TEXT→IMG)
        if backend == "flux" and info and info.get("enabled") and info.get("status") == "ready" and info.get("finetune_id"):
            trigger = info.get("trigger") or ""
            prompt_ft = (f"{trigger}, {prompt}").strip(", ")
            img_bytes = generate_with_flux_finetuned(prompt_ft, info["finetune_id"], finetune_strength=0.9)
        else:
            # Иначе — face-lock или обычная стилизация FLUX с image_prompt
            face_bytes = USER_LAST_PHOTO[uid]
            if backend == "instantid":
                img_bytes = generate_with_instantid(face_bytes, prompt)
            elif backend == "ipadapter":
                img_bytes = generate_with_ipadapter(face_bytes, prompt)
            else:  # обычный FLUX
                # если ты используешь LoRA-URL вместо finetune_id, можно подмешивать сюда:
                lora_url = info.get("url") if info else None
                img_bytes = generate_with_flux(face_bytes, prompt, lora_url=lora_url)

        await bot.send_photo(
            chat_id=uid,
            photo=BufferedInputFile(img_bytes, filename=f"{key}.jpg"),
            caption=f"Готово! Режим: {backend}  |  Стиль: {key}"
        )
    except Exception as e:
        logging.exception("Generation error")
        await cq.message.answer(f"⚠️ Ошибка генерации: {e}\nПопробуй другой стиль/режим или другое фото.")

# ======== FINETUNE/LoRA FLOW ========
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
            "Задай её в Railway → Variables (пример: black-forest-labs/flux-pro-trainer) и повтори."
        )

    USER_LORA[uid] = {"finetune_id": None, "trigger": None, "enabled": False, "status": "training"}
    await m.answer("🚀 Запускаю обучение финетюна… Я напишу, когда будет готово.")
    asyncio.create_task(_train_and_notify(uid, imgs))

@dp.message(Command("lora_on"))
async def lora_on(m: Message):
    uid = m.from_user.id
    info = USER_LORA.get(uid, {})
    # включаем и для finetune_id, и для url-варианта
    if not (info.get("finetune_id") or info.get("url")):
        return await m.answer("LoRA/finetune ещё не задан. Сначала /train или /set_finetune <id>.")
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
