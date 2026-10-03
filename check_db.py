import asyncio
from app.db.database import SessionLocal
from app.db.repository import get_items


async def main():
    items = await get_items(user_id=213102413)  # ← подставьте ваш user_id
    for it in items:
        print(it)


if __name__ == "__main__":
    asyncio.run(main())
