#!/usr/bin/env python3
"""Live autotest for wardrobe_bot (@MyWordroobBot).

Scenario (per image):
  a. send photo to bot
  b. wait for the bot's reply (timeout)
  c. capture reply text + elapsed time (seconds)
  d. send /cancel
  then repeat for every image in clothes_images.zip.

Pre-req:
  * Bot running live:  `python app/main.py`   (uses BOT_TOKEN from .env)
  * A Telegram **user** account for the client that sends photos.
  * Env vars:
      TG_API_ID       (from https://my.telegram.org)
      TG_API_HASH     (from https://my.telegram.org)
      TG_PHONE        (user's phone number, e.g. +1234567890)
      TG_SESSION      (optional: name of the Telethon session file)
      TG_BOT_USERNAME target bot username, without '@'  (default MyWordroobBot)
      TG_ARCHIVE      path to clothes_images.zip
                       (default C:/Users/User/Desktop/Гардероб/clothes_images.zip)

The first run prompts for the Telegram login code in the terminal; the
session is then saved locally so subsequent runs reuse it.
Output: tests/report_<timestamp>.json  +  tests/report_<timestamp>.csv
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from telethon import TelegramClient, errors
from telethon.tl.types import InputPeerUser

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
BOT_USERNAME = os.getenv("TG_BOT_USERNAME", "MyWordroobBot")
SESSION_NAME = os.getenv("TG_SESSION", str(PROJECT_ROOT / "tests" / ".autotest.session"))
REPORT_DIR = PROJECT_ROOT / "tests"

API_ID = os.getenv("TG_API_ID")
API_HASH = os.getenv("TG_API_HASH")
PHONE = os.getenv("TG_PHONE")

# How long to wait for a reply to a photo.
REPLY_TIMEOUT = float(os.getenv("REPLY_TIMEOUT", "60"))
# Small pause between sending photo and reading replies / after cancel.
SETTLE_SECONDS = float(os.getenv("SETTLE_SECONDS", "1.0"))
# Per-iteration timeout overall (in case something hangs).
ITER_TIMEOUT = float(os.getenv("ITER_TIMEOUT", "120"))

if not (API_ID and API_HASH and PHONE):
    raise SystemExit(
        "Set TG_API_ID, TG_API_HASH and TG_PHONE env vars before running.\n"
        "Get api_id/api_hash from https://my.telegram.org"
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Result:
    image: str
    bytes_sent: int
    reply_received: bool
    reply_text: str = ""
    reply_first: bool = False
    elapsed: float | None = None
    cancel_ok: bool = False
    error: str = ""


@dataclass
class Report:
    bot_username: str
    archive: str
    started_at: str
    finished_at: str
    total: int = 0
    replied: int = 0
    results: list[Result] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_images(archive: Path) -> list[bytes]:
    """Return [(name, bytes)] for every image entry in the zip, sorted."""
    with zipfile.ZipFile(archive, "r") as zf:
        names = sorted(
            (n for n in zf.namelist() if not n.endswith("/")),
        )
        out = []
        for n in names:
            out.append((n, zf.read(n)))
        return out


async def wait_reply(client: TelegramClient, entity, after_id: int) -> tuple[bool, str, datetime | None]:
    """Return (got, text, msg_dt) for the first bot message after outgoing id."""
    if isinstance(entity, InputPeerUser):
        # entity is a peer; use the peer directly
        peer = entity
    else:
        peer = entity

    deadline = time.monotonic() + REPLY_TIMEOUT
    async for msg in client.iter_messages(peer, min_id=after_id):
        # Only messages from the bot itself.
        sender = await msg.get_sender()
        if sender is not None and getattr(sender, "bot", False):
            text = (msg.text or "").rstrip()
            return True, text, msg.date
    # iter_messages is not a live stream; poll instead.
    last_seen: set[int] = set()
    while time.monotonic() < deadline:
        async for msg in client.iter_messages(peer, min_id=after_id):
            if msg.id in last_seen:
                continue
            last_seen.add(msg.id)
            sender = await msg.get_sender()
            if sender is not None and getattr(sender, "bot", False):
                text = (msg.text or "").rstrip()
                return True, text, msg.date
        await asyncio.sleep(0.75)
    return False, "", None


async def cancel(client: TelegramClient, peer) -> bool:
    try:
        await client.send_message(peer, "/cancel")
        return True
    except Exception:
        return False


async def run() -> Report:
    images = list_images(ARCHIVE)
    if not images:
        print(f"[WARN] no image entries found in {ARCHIVE}")

    report = Report(
        bot_username=BOT_USERNAME,
        archive=str(ARCHIVE),
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start(phone=PHONE)
    me = await client.get_me()
    print(f"[INFO] logged in as {getattr(me, 'username', me.id)}")

    try:
        bot_entity = await client.get_entity(BOT_USERNAME)
    except (ValueError, errors.UsernameNotOccupiedError) as e:
        print(f"[ERROR] cannot resolve @{BOT_USERNAME}: {e}")
        return report

    # Warm up: /start so the bot registers the user (some graphs require it).
    try:
        await client.send_message(bot_entity, "/start")
        await asyncio.sleep(SETTLE_SECONDS)
    except Exception as e:
        print(f"[WARN] /start failed: {e}")

    report.total = len(images)
    for name, data in images:
        res = Result(image=name, bytes_sent=len(data))
        print(f"[>>] {name} ({len(data)} bytes)")
        t0 = time.monotonic()
        try:
            async with asyncio.timeout(ITER_TIMEOUT):
                sent = await client.send_file(
                    bot_entity,
                    data,
                    caption=name,
                    force_document=False,
                )
                after_id = getattr(sent, "id", 0)
                send_dt = time.monotonic()
                got, text, msg_dt = await wait_reply(client, bot_entity, after_id)
                if got:
                    # elapsed = time between photo send and reply arrival
                    res.reply_received = True
                    res.reply_text = (text[:200] + "…") if len(text) > 200 else text
                    res.reply_first = True
                    res.elapsed = round(time.monotonic() - send_dt, 3)
                    report.replied += 1
                else:
                    res.error = f"no reply within {REPLY_TIMEOUT}s"
                    print(f"[!!] {name}: no reply in {REPLY_TIMEOUT}s")

                # /cancel
                res.cancel_ok = await cancel(client, bot_entity)
                await asyncio.sleep(SETTLE_SECONDS)
        except asyncio.TimeoutError:
            res.error = f"iteration exceeded {ITER_TIMEOUT}s"
            print(f"[!!] {name}: iteration timeout")
        except Exception as e:
            res.error = f"{type(e).__name__}: {e}"
            print(f"[!!] {name}: {res.error}")

        res.elapsed = round(time.monotonic() - t0, 3) if res.elapsed is None else res.elapsed
        report.results.append(res)

    report.finished_at = datetime.now(timezone.utc).isoformat()
    return report


def write_report(report: Report) -> tuple[Path, Path]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"report_{ts}.json"
    csv_path = REPORT_DIR / f"report_{ts}.csv"

    def _json():
        return json.dumps(report, default=asdict, ensure_ascii=False, indent=2)

    json_path.write_text(_json(), encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["image", "bytes_sent", "reply_received", "reply_first",
             "reply_text", "elapsed_s", "cancel_ok", "error"]
        )
        for r in report.results:
            w.writerow(
                [r.image, r.bytes_sent, r.reply_received, r.reply_first,
                 r.reply_text, r.elapsed, r.cancel_ok, r.error]
            )
    return json_path, csv_path


async def main() -> None:
    print("[INFO] extracting archive:", ARCHIVE)
    report = await run()
    jp, cp = write_report(report)
    print("\n===== SUMMARY =====")
    print(f"bot            : @{report.bot_username}")
    print(f"images total   : {report.total}")
    print(f"replied        : {report.replied}")
    print(f"json report    : {jp}")
    print(f"csv report     : {cp}")
    for r in report.results:
        status = "OK" if r.reply_received else "NO-REPLY"
        print(f"  [{status}] {r.image[:40]:<40} elapsed={r.elapsed}s cancel={r.cancel_ok} err={r.error}")


if __name__ == "__main__":
    asyncio.run(main())
