import requests
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
HEADERS_HF = {"Authorization": f"Bearer {HF_TOKEN}"}

def transcribe_with_huggingface(audio_path):
    url = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "audio/ogg"},
        data=open(audio_path, "rb").read()
    )
    response.raise_for_status()
    data = response.json()
    return data.get("text")

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
    audio_path = f"audio.ogg"
    await file.download_to_drive(audio_path)

    try:
        text = transcribe_with_huggingface(audio_path)
    except Exception as e:
        await update.message.reply_text("Не удалось распознать аудио")
        print(e)
        return

    await update.message.reply_text(f"🗣️ Распознанный текст: {text}")

    try:
        answer = generate_response(text)
        await update.message.reply_text(f"🤖 {answer}")
    except Exception as e:
        await update.message.reply_text("Ошибка при генерации ответа")
        print(e)

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.VOICE, voice_handler))
    print("🤖 Бот запущен")
    app.run_polling()


if __name__ == "__main__":
    main()