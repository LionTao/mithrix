import os
import asyncio
import logging

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
from aiogram.types import Message
from transformers import AutoTokenizer, AutoModel
from aiogram.client.session.aiohttp import AiohttpSession

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

PROXY_URL = os.getenv("PROXY_URL")
API_TOKEN = os.getenv("API_TOKEN")
MODEL_PATH = os.getenv("MODEL_PATH")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device='cuda')
model = model.eval()
history = []

session = AiohttpSession(
    proxy={
       PROXY_URL,
    }  # can be any iterable if not set
)
bot = Bot(API_TOKEN, parse_mode="HTML", session=session)

router = Router()

@router.message(Command(commands=["hi","start"]))
async def command_start_handler(message: Message) -> None:
    """
    This handler receive messages with `/start` command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    await message.answer(f"Hello, <b>{message.from_user.full_name}!</b>")

@router.message(Command(commands=["ask"]))
async def chat_handler(message: types.Message) -> None:
    """
    Handler will forward received message back to the sender

    By default, message handler will handle all message types (like text, photo, sticker and etc.)
    """
    global history
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        # Send copy of the received message
        response, history = model.chat(tokenizer, message.text, history=history)
        await message.reply(response)
    except TypeError:
        # But not all the types is supported to be copied so need to handle it
        await message.answer("Nice try!")

@router.message(Command(commands=["flush","clear"]))
async def clear_handler(message: types.Message) -> None:
    """
    Handler will forward received message back to the sender

    By default, message handler will handle all message types (like text, photo, sticker and etc.)
    """
    global history
    try:
        # Send copy of the received message
        history = []
        await message.reply("memory flushed!")
    except TypeError:
        # But not all the types is supported to be copied so need to handle it
        await message.answer("Nice try!")

@router.message()
async def echo_handler(message: types.Message) -> None:
    """
    Handler will forward received message back to the sender

    By default, message handler will handle all message types (like text, photo, sticker and etc.)
    """
    try:
        # Send copy of the received message
        await message.send_copy(chat_id=message.chat.id)
    except TypeError:
        # But not all the types is supported to be copied so need to handle it
        await message.answer("Nice try!")

async def main() -> None:
    # Dispatcher is a root router
    dp = Dispatcher()
    # ... and all other routers should be attached to Dispatcher
    dp.include_router(router)
    # And the run events dispatching
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
