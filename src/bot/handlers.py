import os
import aiohttp

from aiogram import Router
from aiogram.types import Message
from aiogram.filters import CommandStart

from .keyboards import start_keyboard
from .config import API_URL

router = Router()


@router.message(CommandStart())
async def start(message: Message):

    text = (
        "👋 Привет!\n\n"
        "Я бот для распознавания мировых достопримечательностей.\n\n"
        "📷 Нажми кнопку ниже и отправь фотографию."
    )

    await message.answer(text, reply_markup=start_keyboard)


@router.message(lambda message: message.text == "📷 Отправить фото")
async def ask_photo(message: Message):

    text = (
        "Отлично! 📷\n\n"
        "Отправь фотографию достопримечательности,\n"
        "и я попробую её распознать."
    )

    await message.answer(text)


@router.message(lambda message: message.photo)
async def handle_photo(message: Message, bot):

    wait_msg = await message.answer(
        "🔎 Анализирую изображение...\n"
        "Это может занять пару секунд."
    )

    photo = message.photo[-1]

    file = await bot.get_file(photo.file_id)

    os.makedirs("temp", exist_ok=True)

    file_path = f"temp/{photo.file_id}.jpg"

    await bot.download_file(file.file_path, file_path)

    async with aiohttp.ClientSession() as session:

        with open(file_path, "rb") as f:

            data = aiohttp.FormData()
            data.add_field("file", f, filename="image.jpg")

            async with session.post(API_URL + "/predict", data=data) as resp:

                result = await resp.json()

    text = (
        f"📍 **Результат распознавания**\n\n"
        f"🏛 Место: {result['landmark']}\n"
        f"📊 Вероятность: {result['probability']:.2%}\n\n"
        f"📝 Описание:\n{result['description']}"
    )

    await wait_msg.edit_text(text)

    os.remove(file_path)