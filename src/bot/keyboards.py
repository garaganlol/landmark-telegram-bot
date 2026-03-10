from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

start_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📷 Отправить фото")],
        [KeyboardButton(text="ℹ️ О боте")]
    ],
    resize_keyboard=True
)