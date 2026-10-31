"""Замер скорости распознавания на тестовых фото."""
import asyncio
import time
from pathlib import Path

from app.services.vision import classify_clothing, warmup


async def main():
    print("Прогрев моделей...")
    t0 = time.time()
    warmup()
    print(f"Прогрев занял {time.time() - t0:.1f} сек\n")

    paths = sorted(Path("/app").glob("test*.jpeg"))
    if not paths:
        paths = sorted(Path("/app").glob("test*.jpg"))

    times = []
    for p in paths:
        with open(p, "rb") as f:
            data = f.read()
        t = time.time()
        r = await classify_clothing(data)
        dt = time.time() - t
        times.append(dt)
        print(
            f"{p.name:30s} | {r.get('type', '?'):15s} "
            f"| conf={r.get('confidence', 0):.2f} "
            f"| {r.get('color', '?'):10s} "
            f"| {r.get('material', '?') or '—':10s} "
            f"| {dt:.2f} сек"
        )

    if times:
        print(f"\nСреднее время: {sum(times)/len(times):.2f} сек")
        print(f"Минимум: {min(times):.2f} сек, максимум: {max(times):.2f} сек")


if __name__ == "__main__":
    asyncio.run(main())
