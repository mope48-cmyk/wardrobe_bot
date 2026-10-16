#!/usr/bin/env python3
"""Live autotest for wardrobe_bot using Bot API (no Telethon needed).

Uses the BOT_TOKEN from .env to send photos and receive updates via getUpdates.
Scenario per image:
  a. send photo to @MyWordroobBot
  b. wait for bot's reply (poll getUpdates with timeout)
  c. capture reply text + elapsed time
  d. send /cancel
  repeat for every image in clothes_images.zip.

Set env vars (or they'll be read from .env):
  BOT_TOKEN       - from .env (already known)
  TG_ARCHIVE      - path to clothes_images.zip (default C:/Users/User/Desktop/Гардероб/clothes_images.zip)
  REPLY_TIMEOUT   - seconds to wait for reply (default 60)
  ITER_TIMEOUT    - per-iteration overall timeout (default 120)
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = Path(
    os.getenv(
        "TG_ARCHIVE",
        r"C:\Users\User\Desktop\Гардероб\clothes_images.zip",
    )
)
BOT_TOKEN = os.getenv(
    "BOT_TOKEN",
    None,
)  # will fall back to reading .env
REPLY_TIMEOUT = float(os.getenv("REPLY_TIMEOUT", "60"))
ITER_TIMEOUT = float(os.getenv("ITER_TIMEOUT", "120"))

# Read token from .env if not provided
if not BOT_TOKEN:
    env_path = Path("C:/Projects/wardrobe_bot/.env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("BOT_TOKEN") and "=" in line:
                BOT_TOKEN = line.split("=", 1)[1].strip().strip('"')
                break

if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN not set. Put it in env or .env.")

BASE_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
ARCHIVE_IMAGES: list[tuple[str, bytes]] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_images(archive: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(archive, "r") as zf:
        names = sorted(n for n in zf.namelist() if not n.endswith("/"))
        out = []
        for n in names:
            out.append((n, zf.read(n)))
        return out


async def send_photo(bot_token: str, chat_id: int, photo_bytes: bytes, caption: str | None = None) -> dict:
    """Send photo via Bot API sendPhoto. Returns API response dict."""
    async with httpx.AsyncClient(timeout=ITER_TIMEOUT) as client:
        resp = await client.post(
            f"{BASE_API}/sendPhoto",
            data={"chat_id": chat_id, "caption": caption or "",
                  "photo": ("photo.jpg", photo_bytes, "image/jpeg")},
        )
        resp.raise_for_status()
        return resp.json()


async def send_message(bot_token: str, chat_id: int, text: str = "/cancel") -> dict:
    async with httpx.AsyncClient(timeout=ITER_TIMEOUT) as client:
        resp = await client.post(
            f"{BASE_API}/sendMessage",
            data={"chat_id": chat_id, "text": text},
        )
        resp.raise_for_status()
        return resp.json()


async def get_updates(offset: int | None = None, timeout: int = REPLY_TIMEOUT) -> list[dict]:
    """Poll getUpdates. Returns list of update dicts (or empty)."""
    async with httpx.AsyncClient(timeout=ITER_TIMEOUT) as client:
        params = {"timeout": min(timeout, 30)}  # Telegram max 30s per request
        if offset is not None:
            params["offset"] = offset
        resp = await client.get(f"{BASE_API}/getUpdates", params=params)
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            return data["result"]
        return []


def find_bot_response(updates: list[dict], my_message_id: int | None = None) -> dict | None:
    """Find the bot's reply in the updates list.
    
    Heuristic: look for a message that:
    - is in the same chat as my sent message
    - has a sender marked as bot=True
    - comes after my message id
    - has text content
    """
    for up in updates:
        if not up.get("message"):
            continue
        msg = up["message"]
        # Must be a bot sender
        if not msg.get("from", {}).get("is_bot"):
            continue
        # Must have text
        if not msg.get("text"):
            continue
        # Ideally comes after our message; we'll accept the first bot text message
        return msg
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    global ARCHIVE_IMAGES
    ARCHIVE_IMAGES = list_images(ARCHIVE)
    if not ARCHIVE_IMAGES:
        raise SystemExit(f"No images in archive: {ARCHIVE}")

    report = {
        "bot_username": "MyWordroobBot",
        "archive": str(ARCHIVE),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "total": len(ARCHIVE_IMAGES),
        "replied": 0,
        "results": [],
    }

    # We'll keep track of the last update offset to avoid re-processing
    # But since we send photos sequentially, we need a fresh offset each time
    # Strategy: send photo, then poll getUpdates from the last known offset+1

    last_offset = 0

    for name, data in ARCHIVE_IMAGES:
        if len(data) > 10*1024*1024:
            res = {"image":name,"bytes_sent":len(data),"reply_received":False,"reply_text":"","elapsed":0.0,"cancel_ok":False,"error":"skipped (>10MB)"}
            report["results"].append(res); print(f"    [SKIP] {name}"); await asyncio.sleep(0.5); continue
        res = {
            "image": name,
            "bytes_sent": len(data),
            "reply_received": False,
            "reply_text": "",
            "elapsed": None,
            "cancel_ok": False,
            "error": "",
        }
        print(f"[>>] {name} ({len(data)} bytes)")

        t0 = time.monotonic()

        try:
            async with asyncio.timeout(ITER_TIMEOUT):
                # 1. Send photo to bot
                # We need a chat_id. Since we don't know it ahead, we'll use getUpdates
                # to discover the chat, or we can send to the bot's username.
                # Actually, sendPhoto requires chat_id (numeric). We can resolve via
                # getUpdates or by sending to the bot and using the returned message.
                # Strategy: send photo, the API returns message with chat_id.
                
                api_resp = await send_photo(BOT_TOKEN, 0, data, caption=name)
                if not api_resp.get("ok"):
                    raise RuntimeError(f"sendPhoto failed: {api_resp}")
                
                sent_msg = api_resp["result"]
                chat_id = sent_msg["chat"]["id"]
                my_msg_id = sent_msg["message_id"]
                print(f"    Sent to chat {chat_id}, message_id {my_msg_id}")

                # 2. Poll getUpdates for bot's reply
                deadline = time.monotonic() + REPLY_TIMEOUT
                bot_reply = None
                while time.monotonic() < deadline:
                    updates = await get_updates(offset=last_offset + 1, timeout=5.0)
                    for up in updates:
                        last_offset = up["update_id"]  # advance offset
                        msg = find_bot_response(updates, my_msg_id)
                        if msg:
                            bot_reply = msg
                            break
                    if bot_reply:
                        break
                    # small sleep to not hammer
                    await asyncio.sleep(0.5)

                if not bot_reply:
                    res["error"] = f"no bot reply within {REPLY_TIMEOUT}s"
                    print(f"    [!!] {res['error']}")
                else:
                    # 3. Record response
                    reply_text = (bot_reply.get("text") or "").strip()
                    res["reply_received"] = True
                    res["reply_text"] = reply_text[:200] + ("…" if len(reply_text) > 200 else "")
                    # Elapsed: time from sending photo until reply arrived
                    res["elapsed"] = round(time.monotonic() - t0, 3)
                    report["replied"] += 1
                    print(f"    <- Reply: {reply_text[:80]}... elapsed={res['elapsed']}s")

                # 4. Send /cancel
                cancel_resp = await send_message(BOT_TOKEN, chat_id, "/cancel")
                res["cancel_ok"] = cancel_resp.get("ok", False)
                await asyncio.sleep(1.0)  # settle
                print(f"    /cancel sent: {cancel_resp}")

        except asyncio.TimeoutError:
            res["error"] = f"iteration exceeded {ITER_TIMEOUT}s"
            print(f"    [!!] {res['error']}")
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {e}"
            print(f"    [!!] {res['error']}")

        res["elapsed"] = round(time.monotonic() - t0, 3) if res.get("elapsed") is None else res["elapsed"]
        report["results"].append(res)

    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    # Save reports
    REPORT_DIR = PROJECT_ROOT / "tests"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = REPORT_DIR / f"report_{ts}.json"
    csv_path = REPORT_DIR / f"report_{ts}.csv"

    # JSON
    json_content = json.dumps(report, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else o, ensure_ascii=False, indent=2)
    json_path.write_text(json_content, encoding="utf-8")

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "bytes_sent", "reply_received", "reply_text", "elapsed_s", "cancel_ok", "error"])
        for r in report["results"]:
            w.writerow([r["image"], r["bytes_sent"], r["reply_received"], r["reply_text"], r["elapsed"], r["cancel_ok"], r["error"]])

    print("\n===== SUMMARY =====")
    print(f"bot            : @MyWordroobBot")
    print(f"images total   : {report['total']}")
    print(f"replied        : {report['replied']}")
    print(f"json report    : {json_path}")
    print(f"csv report     : {csv_path}")
    for r in report["results"]:
        status = "OK" if r["reply_received"] else "NO-REPLY"
        print(f"  [{status}] {r['image'][:30]:<30} elapsed={r['elapsed']}s cancel={r['cancel_ok']} err={r['error']}")


if __name__ == "__main__":
    asyncio.run(main())