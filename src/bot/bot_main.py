import asyncio

from aiogram import Bot, Dispatcher

from src.bot.config import BOT_TOKEN
from src.bot.handlers import router

async def main():

    bot = Bot(token=BOT_TOKEN)

    dp = Dispatcher()

    dp.include_router(router)

    await bot.delete_webhook(drop_pending_updates=True)

    print("Bot started")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())