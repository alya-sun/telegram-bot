import requests
import subprocess
import sys
import os
import re
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

API_KEY = os.getenv("API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def run_whisper(audio_path):
    cmd = [
        "python", "python-clients/scripts/asr/transcribe_file_offline.py",
        "--server", "grpc.nvcf.nvidia.com:443",
        "--use-ssl",
        "--metadata", "function-id", "b702f636-f60c-4a3d-a6f4-f3568c13bd7d",
        "--metadata", "authorization", f"Bearer {API_KEY}",
        "--language-code", "multi",
        "--input-file", audio_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    if result.returncode != 0:
        print("⚠️ Ошибка:", result.stderr)
        return None
    return result.stdout.strip()


def extract_transcript(raw_output):
    match = re.search(r'transcript:\s*"(.+?)"', raw_output)
    return match.group(1).strip() if match else None

def generate_response(prompt):
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "google/gemma-3n-e2b-it",
        "parameters": {
            "max_new_tokens": 250,
            "top_p": 1,
            "temperature": 0.7
        }
    }
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    output = response.json()
    return output["choices"][0]["message"]["content"]

async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    ogg_path = f"audio.ogg"
    await file.download_to_drive(ogg_path)

    raw_output = run_whisper(ogg_path)
    text = extract_transcript(raw_output)
    if not text:
        await update.message.reply_text("😕 Не удалось распознать аудио.")
        return

    await update.message.reply_text(f"🗣️ Распознанный текст: {text}")

    try:
        answer = generate_response(text)
        await update.message.reply_text(f"🤖 {answer}")
    except Exception as e:
        await update.message.reply_text("⚠️ Ошибка при генерации ответа.")
        print(e)

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.VOICE, voice_handler))
    print("🤖 Бот запущен")
    app.run_polling()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    main()