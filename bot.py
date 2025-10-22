import os
import json
import uuid
import tempfile
import logging
import requests
import openai
import redis
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.utils.exceptions import BotBlocked
from fastapi import FastAPI
from starlette.requests import Request
import uvicorn
import time
from pathlib import Path

# 🔹 Загружаем переменные окружения
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")
TTS_PROVIDER = os.getenv("TTS_PROVIDER")  # "OPENAI" или "ELEVENLABS"
MODEL_CHAT = os.getenv("MODEL_CHAT")
MODEL_EVAL = os.getenv("MODEL_EVAL")
BOT_PASSWORD = os.getenv("BOT_PASSWORD")

# 🔹 Настройки Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 1))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# 🔹 Настройки OpenAI API
openai.api_key = OPENAI_API_KEY
# openai.base_url = "https://api.deepseek.com"

# 🔹 Прокси для OpenAI (если нужно)
PROXY_URL = "http://user206026:yxj7mc@82.153.34.148:9297"
os.environ["HTTPS_PROXY"] = PROXY_URL
os.environ["HTTP_PROXY"] = PROXY_URL

# 🔹 Параметры организации и БД
ORG_ID = os.getenv("ORG_ID", "5")
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "dialog_analysis"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "charset": "utf8mb4",
    "use_unicode": True,
    "ssl_disabled": True,
}
EVAL_LIMIT_PER_WEEK = int(os.getenv("EVAL_LIMIT_PER_WEEK", "3"))
REPORT_CRITERION = os.getenv("REPORT_CRITERION", "evaluation_report")

# 🔹 Настройки логирования
logging.basicConfig(level=logging.INFO)

# 🔹 Инициализация бота и диспетчера
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# 🔹 Глобальный обработчик ошибок
@dp.errors_handler()
async def errors_handler(update, exception):
    if isinstance(exception, BotBlocked):
        logging.warning(f"Ignored BotBlocked for update: {update}")
        return True
    return False

# 🔹 FastAPI приложение
app = FastAPI()

# 🔹 Функция загрузки промпта
def load_prompt(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

PROMPT = load_prompt("prompt_stas.txt")
PROMPT_SUPERVISOR = None  # Промпт для оценки загружается из БД

# 🔹 Работа с базой данных
def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        logging.error(f"DB connection error: {e}")
        return None

def load_criteria(conn):
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, name, type FROM audit_criteria WHERE organization_id=%s AND is_active=1 ORDER BY id",
            (ORG_ID,),
        )
        rows = cur.fetchall()
        cur.close()
        return rows
    except Error as e:
        logging.error(f"Failed to load criteria: {e}")
        return []

def get_sale_points(conn):
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT sale_point_id, display_name FROM sale_points WHERE organization_id=%s AND is_active=1 ORDER BY sale_point_id",
            (ORG_ID,),
        )
        rows = cur.fetchall()
        cur.close()
        return rows
    except Error as e:
        logging.error(f"Failed to load sale points: {e}")
        return []


def load_supervisor_prompt(conn):
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT prompt_id, prompt_text FROM analysis_prompts WHERE organization_id = %s AND is_active = 1 ORDER BY version DESC LIMIT 1",
            (ORG_ID,),
        )
        row = cur.fetchone()
        cur.close()
        if row:
            return row["prompt_id"], row["prompt_text"]
    except Error as e:
        logging.error(f"Failed to load supervisor prompt: {e}")
    return None, None


# 🔹 Проверка и сохранение авторизации пользователя
def is_authorized(chat_id):
    return redis_client.get(f"auth:{chat_id}") == "1"


def set_authorized(chat_id):
    redis_client.set(f"auth:{chat_id}", "1")


def awaiting_password(chat_id):
    return redis_client.get(f"await_pass:{chat_id}") == "1"


def set_await_password(chat_id, value=True):
    if value:
        redis_client.set(f"await_pass:{chat_id}", "1")
    else:
        redis_client.delete(f"await_pass:{chat_id}")


# 🔹 Параметры для истории чата
# Максимальное количество сообщений (пользователь + ассистент) для сохранения.
# Системное сообщение (промпт) всегда остается.
MAX_USER_MESSAGES = 50

# 🔹 Функции для работы с историей диалога в Redis

def get_conversation_key(chat_id):
    """Возвращает ключ Redis для истории данного чата."""
    return f"chat_history:{chat_id}"

def init_conversation_history(chat_id):
    """
    Инициализирует историю чата с системным промптом.
    Системное сообщение сохраняется с ролью "system".
    """
    key = get_conversation_key(chat_id)
    system_message = {"role": "system", "content": PROMPT}
    redis_client.delete(key)  # Очищаем историю, если она уже существует
    redis_client.rpush(key, json.dumps(system_message))
    return [system_message]

def get_conversation_history(chat_id):
    """
    Возвращает историю чата для данного chat_id.
    Если истории нет, то инициализирует её.
    """
    key = get_conversation_key(chat_id)
    history = redis_client.lrange(key, 0, -1)
    if not history:
        return init_conversation_history(chat_id)
    return [json.loads(item) for item in history]

def trim_conversation_history(chat_id, max_user_messages=MAX_USER_MESSAGES):
    """
    Оставляет в истории системное сообщение (первый элемент) и последние max_user_messages сообщений.
    """
    key = get_conversation_key(chat_id)
    history = get_conversation_history(chat_id)
    # Общее число сообщений должно быть не более: 1 (системный) + max_user_messages
    if len(history) > max_user_messages + 1:
        new_history = [history[0]] + history[-max_user_messages:]
        redis_client.delete(key)
        for msg in new_history:
            redis_client.rpush(key, json.dumps(msg))

def append_message(chat_id, role, content):
    """
    Добавляет сообщение в историю чата и подрезает историю при необходимости.
    """
    key = get_conversation_key(chat_id)
    message = {"role": role, "content": content}
    redis_client.rpush(key, json.dumps(message))
    trim_conversation_history(chat_id, MAX_USER_MESSAGES)

# 🔹 Функция выбора генерации озвучки
def generate_audio(text):
    if TTS_PROVIDER.upper() == "OPENAI":
        return generate_audio_openai(text)
    elif TTS_PROVIDER.upper() == "ELEVENLABS":
        return generate_audio_elevenlabs(text)
    else:
        raise ValueError("Неверное значение TTS_PROVIDER. Доступные варианты: 'OPENAI', 'ELEVENLABS'.")

# 🔹 Генерация аудио через OpenAI
def generate_audio_openai(text):

    speech_file_path = Path(tempfile.mktemp(suffix=".mp3"))

    with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="ballad",
        input=text,
        instructions="Speak in Russian with a neutral Moscow accent. Be warm, friendly, slightly enthusiastic. Keep speech fast (160-180 wpm). Intonation should be gently rising on questions to show interest. Smile with your voice. Do not whisper.",
    ) as response:
        response.stream_to_file(speech_file_path)

    return speech_file_path

# 🔹 Генерация аудио через ElevenLabs с использованием прокси
def generate_audio_elevenlabs(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    proxies = {
        "http": PROXY_URL,
        "https": PROXY_URL,
    }
    
    response = requests.post(url, headers=headers, json=data, proxies=proxies)
    
    if response.status_code != 200:
        raise Exception(f"Ошибка при вызове ElevenLabs API: {response.status_code} - {response.text}")
    
    temp_audio_path = tempfile.mktemp(suffix=".mp3")
    with open(temp_audio_path, "wb") as f:
        f.write(response.content)
    
    return temp_audio_path

# 🔹 Функция оценки диалога
def evaluate_conversation(chat_id, sale_point_id, start_time_str):
    history = get_conversation_history(chat_id)

    if len(history) <= 1:
        return "Нет данных для оценки. Начните разговор с ботом."

    user_assistant_history = history[1:]
    dialog_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in user_assistant_history]
    )
    logging.info(
        "Оценивается диалог для chat_id %s. История: %s",
        chat_id,
        dialog_text,
    )

    conn = get_db_connection()
    if not conn:
        return "Ошибка подключения к базе данных"

    prompt_id, prompt_text = load_supervisor_prompt(conn)
    if not prompt_text:
        conn.close()
        return "Не найден активный промпт для оценки"

    criteria = load_criteria(conn)
    criteria_map = {c["name"]: c for c in criteria}

    properties = {}
    required = []
    for crit in criteria:
        required.append(crit["name"])
        if crit["type"] == "string":
            properties[crit["name"]] = {"type": "string"}
        elif crit["type"] == "array_string":
            properties[crit["name"]] = {"type": "array", "items": {"type": "string"}}
        elif crit["type"] == "float":
            properties[crit["name"]] = {"type": "number"}
        else:
            properties[crit["name"]] = {"type": "integer"}

    def analyze_dialog(**kwargs):
        tz = pytz.timezone("Europe/Moscow")
        start_dt = datetime.fromisoformat(start_time_str).astimezone(tz)
        end_dt = datetime.now(tz)
        filename = f"bot_dialog_chunk_{chat_id}_{start_dt.strftime('%Y%m%d%H%M%S')}_{end_dt.strftime('%Y%m%d%H%M%S')}"
        chunk_date = start_dt.date()

        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO dialog_chunks (date, start_datetime, end_datetime, filename, organization_id, sale_point_id) VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    chunk_date,
                    start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    end_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    filename,
                    ORG_ID,
                    sale_point_id,
                ),
            )
            chunk_id = cur.lastrowid

            os.makedirs("dialogs", exist_ok=True)
            with open(os.path.join("dialogs", f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write(dialog_text)

            cur.execute(
                "INSERT INTO analysis_versions (description, prompt_id) VALUES (%s,%s)",
                (
                    f"Sale_point: {sale_point_id} Chat_id: {chat_id} Bot-Stas: yes",
                    prompt_id,
                ),
            )
            version_id = cur.lastrowid

            cur.execute(
                "INSERT INTO dialog_analysis (chunk_id, version_id) VALUES (%s,%s)",
                (chunk_id, version_id),
            )
            analysis_id = cur.lastrowid

            for name, value in kwargs.items():
                crit = criteria_map.get(name)
                if not crit:
                    continue
                cur.execute(
                    "INSERT INTO dialog_analysis_values (analysis_id, criterion_id, value_text) VALUES (%s,%s,%s)",
                    (analysis_id, crit["id"], json.dumps(value, ensure_ascii=False)),
                )

            conn.commit()
        except Error as e:
            logging.error(f"Failed to save analysis: {e}")
        finally:
            cur.close()
            conn.close()
        return kwargs

    tools = [
        {
            "type": "function",
            "function": {
                "name": "analyze_dialog",
                "description": "Сохраняет результаты анализа",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]

    args = None
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        completion = openai.chat.completions.create(
            model=MODEL_EVAL,
            messages=[{"role": "system", "content": prompt_text}, {"role": "user", "content": dialog_text}],
            tools=tools,
        )

        tool_calls = getattr(completion.choices[0].message, "tool_calls", []) or []
        if tool_calls:
            tool_call = tool_calls[0]
            try:
                args = json.loads(tool_call.function.arguments)
                break
            except json.JSONDecodeError as e:
                logging.error(
                    "Attempt %s: failed to decode JSON from LLM response: %s. Arguments: %s",
                    attempt,
                    e,
                    tool_call.function.arguments,
                )
                if attempt == max_attempts:
                    logging.error("Max attempts reached while decoding JSON")
                    return "В процессе оценки диалога произошла ошибка, обратитесь к разработчику"
                continue
        else:
            args = {name: 0 for name in required}
            break

    if args is None:
        args = {name: 0 for name in required}

    result = analyze_dialog(**args)
    return result.get(REPORT_CRITERION, "Оценка сохранена")

@dp.message_handler(commands=['start'])
async def handle_start(message: types.Message):
    chat_id = message.chat.id

    if not is_authorized(chat_id):
        set_await_password(chat_id, True)
        await message.answer("Введите пароль для доступа:")
        return

    conn = get_db_connection()
    if not conn:
        await message.answer("Ошибка подключения к базе данных")
        return
    sale_points = get_sale_points(conn)
    conn.close()

    if not sale_points:
        await message.answer("Не удалось получить список филиалов")
        return

    keyboard = types.InlineKeyboardMarkup()
    for sp in sale_points:
        keyboard.add(types.InlineKeyboardButton(sp["display_name"], callback_data=f"sp:{sp['sale_point_id']}"))

    await message.answer("Выберите филиал пиццерии:", reply_markup=keyboard)


@dp.message_handler(lambda message: awaiting_password(message.chat.id), content_types=types.ContentType.TEXT)
async def handle_password(message: types.Message):
    chat_id = message.chat.id
    if message.text.strip() == (BOT_PASSWORD or ""):
        set_authorized(chat_id)
        set_await_password(chat_id, False)
        await message.answer("Пароль принят!")
        await handle_start(message)
    else:
        await message.answer("Неверный пароль. Попробуйте еще раз:")

@dp.callback_query_handler(lambda c: c.data.startswith('sp:'))
async def process_sale_point(callback_query: types.CallbackQuery):
    chat_id = callback_query.message.chat.id
    if not is_authorized(chat_id):
        await bot.answer_callback_query(callback_query.id)
        await bot.send_message(chat_id, "Введите пароль командой /start")
        return
    sale_point_id = callback_query.data.split(':')[1]
    redis_client.set(f"sale_point:{chat_id}", sale_point_id)
    tz = pytz.timezone("Europe/Moscow")
    redis_client.set(f"start_time:{chat_id}", datetime.now(tz).isoformat())
    redis_client.delete(f"evaluated:{chat_id}")
    init_conversation_history(chat_id)
    await bot.answer_callback_query(callback_query.id)
    await bot.send_message(chat_id, "Все готово, чтобы начать новый диалог! Напомню, что я покупатель, который зашел в магазин разливного пива. А вы продавец. Итак, произнесите вашу первую фразу (вы можете писать текстом или записывать голосовые сообщения).")


# 🔹 Фолбэк для необработанных callback-запросов
@dp.callback_query_handler(lambda c: True)
async def fallback_callback_handler(callback_query: types.CallbackQuery):
    await callback_query.answer()

# 🔹 Обработчик команды /evaluate
@dp.message_handler(commands=["evaluate"])
async def handle_evaluate(message: types.Message):
    chat_id = message.chat.id
    if not is_authorized(chat_id):
        await message.answer("Введите пароль командой /start")
        return
    sale_point_id = redis_client.get(f"sale_point:{chat_id}")
    start_time = redis_client.get(f"start_time:{chat_id}")
    if not sale_point_id or not start_time:
        await message.answer("Начните новый диалог командой /start")
        return

    if redis_client.get(f"evaluated:{chat_id}"):
        await message.answer("Диалог уже был оценен. Начните новый диалог командой /start")
        return

    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        week_ago = datetime.utcnow() - timedelta(days=7)
        cursor.execute(
            "SELECT COUNT(*) FROM analysis_versions WHERE description LIKE %s AND created_at >= %s AND is_active = 1",
            (f"%Chat_id: {chat_id}%", week_ago.strftime("%Y-%m-%d %H:%M:%S")),
        )
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        if count >= EVAL_LIMIT_PER_WEEK:
            await message.answer(f"Превышен недельный лимит попыток ({EVAL_LIMIT_PER_WEEK})")
            return

    await message.answer("⏳ Оцениваю диалог...")
    evaluation = evaluate_conversation(chat_id, sale_point_id, start_time)
    redis_client.set(f"evaluated:{chat_id}", "1")
    await message.answer(f"📊 Оценка диалога:\n\n{evaluation}")

# 🔹 Обработчик текстовых сообщений
@dp.message_handler(content_types=types.ContentType.TEXT)
async def handle_text(message: types.Message):
    chat_id = message.chat.id
    if awaiting_password(chat_id):
        await handle_password(message)
        return
    if not is_authorized(chat_id):
        await message.answer("Введите пароль командой /start")
        return
    user_input = message.text
    sale_point_id = redis_client.get(f"sale_point:{chat_id}")
    if not sale_point_id:
        await message.answer("Пожалуйста, начните новый диалог командой /start и выберите филиал")
        return
    await message.answer("⏳ Думаю...")
    
    # Инициализируем историю (если ранее не было сообщений) и добавляем новое сообщение пользователя
    _ = get_conversation_history(chat_id)  # инициализация при необходимости
    append_message(chat_id, "user", user_input)
    history = get_conversation_history(chat_id)

    logging.info("History before LLM request for chat %s: %s", chat_id, json.dumps(history, ensure_ascii=False))

    # start_time = time.perf_counter()
    response = openai.chat.completions.create(
        model=MODEL_CHAT,
        messages=history
    )
    #elapsed_time = time.perf_counter() - start_time

    answer = response.choices[0].message.content
    
    # Добавляем информацию о времени выполнения запроса к ответу
    #full_answer = f"{answer}\n\n(Запрос к OpenAI занял {elapsed_time:.2f} секунд)"

    await message.answer(answer)
    # await message.answer(full_answer)
    append_message(chat_id, "assistant", answer)
    
    # Генерация озвучки ответа
    audio_path = generate_audio(answer)
    try:
        with open(audio_path, "rb") as audio:
            await message.answer_voice(audio)
    finally:
        # Удаляем временный файл сразу после использования
        os.remove(audio_path)

# 🔹 Обработчик голосовых сообщений (Speech-to-Text)
@dp.message_handler(content_types=types.ContentType.VOICE)
async def handle_voice(message: types.Message):
    chat_id = message.chat.id
    if awaiting_password(chat_id):
        await message.answer("Введите пароль сначала")
        return
    if not is_authorized(chat_id):
        await message.answer("Введите пароль командой /start")
        return
    sale_point_id = redis_client.get(f"sale_point:{chat_id}")
    if not sale_point_id:
        await message.answer("Пожалуйста, начните новый диалог командой /start и выберите филиал")
        return
    file_info = await bot.get_file(message.voice.file_id)
    file_path = file_info.file_path
    file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
    
    # Создаем уникальное имя файла для временного хранения
    temp_dir = tempfile.gettempdir()
    unique_filename = f"{uuid.uuid4()}.ogg"
    temp_audio_path = os.path.join(temp_dir, unique_filename)
    
    # Скачиваем аудиофайл
    response = requests.get(file_url)
    with open(temp_audio_path, "wb") as f:
        f.write(response.content)
    
    # Распознаем голос
    with open(temp_audio_path, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    
    # Удаляем временный .ogg файл
    os.remove(temp_audio_path)
    
    user_text = transcript.text
    await message.answer(f"Вы сказали: {user_text}")
    
    # Инициализируем историю (если ранее не было сообщений) и добавляем новое сообщение пользователя
    _ = get_conversation_history(chat_id)  # инициализация при необходимости
    # Добавляем сообщение пользователя и обновляем историю
    append_message(chat_id, "user", user_text)
    history = get_conversation_history(chat_id)
    
    logging.info("History before LLM request for chat %s: %s", chat_id, json.dumps(history, ensure_ascii=False))

    response = openai.chat.completions.create(
        model=MODEL_CHAT,
        messages=history
    )
    
    answer = response.choices[0].message.content
    await message.answer(answer)
    append_message(chat_id, "assistant", answer)
    
    # Генерация озвучки ответа
    audio_path = generate_audio(answer)
    try:
        with open(audio_path, "rb") as audio:
            await message.answer_voice(audio)
    finally:
        os.remove(audio_path)

# 🔹 Обработчик вебхука
@app.post("/webhook_soh_stas_bot")
async def webhook(request: Request):
    update = await request.json()
    telegram_update = types.Update(**update)
    await dp.process_update(telegram_update)
    return {"status": "ok"}

# 🔹 Настройка webhook
async def on_startup():
    webhook_url = "https://myuninet.ru/webhook_soh_stas_bot"
    await bot.set_webhook(webhook_url)

async def on_shutdown():
    await bot.delete_webhook()

# 🔹 Запуск FastAPI + aiogram в одном процессе
if __name__ == "__main__":
    import asyncio
    Bot.set_current(bot)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(on_startup())
    uvicorn.run(app, host="0.0.0.0", port=8005)
