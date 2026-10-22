"""Middleware для перехвата альбомов (медиагрупп) в Telegram."""

import asyncio
from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware
from aiogram.types import Message


class AlbumMiddleware(BaseMiddleware):
    """Собирает фото из альбома в один список."""
    
    latency = 0.5  # Ждем полсекунды, чтобы собрать все фото альбома
    album_data: dict[str, list[Message]] = {}

    async def __call__(
        self,
        handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: Dict[str, Any]
    ) -> Any:
        # Если это не альбом — просто пропускаем дальше
        if not event.media_group_id:
            data["album"] = [event]
            return await handler(event, data)

        # Если это альбом — собираем сообщения
        try:
            self.album_data[event.media_group_id].append(event)
            return  # Ждем остальные фото
        except KeyError:
            self.album_data[event.media_group_id] = [event]
            await asyncio.sleep(self.latency)
            # Отдаем весь собранный альбом в handler
            data["album"] = self.album_data.pop(event.media_group_id)
            return await handler(event, data)
