"""Live test: can a Telegram Promote/Keep tap reach this terminal?

This is the same button path ``src.finetune_latest`` uses after a beating run.
It does **not** copy ``best_model.zip`` — it only proves the callback is heard.

Run from the repo root (so ``.env`` is found):

    python test/tg-promote-button-test.py
    python test/tg-promote-button-test.py --wait 180

Then open Telegram and tap **Promote to best_model** or **Keep current**.
The process blocks until you tap, or until ``--wait`` seconds elapse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from src.utils import (  # noqa: E402
    TELEGRAM_CALLBACK_KEEP,
    TELEGRAM_CALLBACK_PROMOTE,
    send_telegram_alert,
    telegram_auth,
    wait_telegram_callback,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Telegram Promote/Keep button round-trip")
    p.add_argument(
        "--wait",
        type=int,
        default=120,
        metavar="SECONDS",
        help="How long to listen for a tap (default: 120).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    auth = telegram_auth()
    if auth is None:
        print("FAIL: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing in .env")
        print("      Same vars inference and finetune use. Nothing was sent.")
        return 2

    token, chat_id = auth
    print("Telegram credentials found.")
    print(f"  chat_id = {chat_id}")
    print(f"  token   = {token[:8]}…{token[-4:]}")
    print()
    print("Sending Promote / Keep buttons (same callback_data as finetune_latest)…")

    markup = {
        "inline_keyboard": [
            [
                {"text": "Promote to best_model", "callback_data": TELEGRAM_CALLBACK_PROMOTE},
                {"text": "Keep current", "callback_data": TELEGRAM_CALLBACK_KEEP},
            ]
        ]
    }
    sent = send_telegram_alert(
        "AirAire BUTTON TEST (not a real promote)\n"
        "Tap Promote or Keep. The GPU/PC terminal should print which button arrived.\n"
        "best_model.zip will NOT be changed by this test.",
        reply_markup=markup,
    )
    if not sent:
        print("FAIL: sendMessage did not succeed. Check bot token / chat id / network.")
        return 1

    print("Message sent. Open Telegram on your phone now.")
    print(f"Listening for up to {args.wait}s … (Ctrl+C to abort)")
    print()

    try:
        choice = wait_telegram_callback(
            timeout_seconds=args.wait,
            allowed=(TELEGRAM_CALLBACK_PROMOTE, TELEGRAM_CALLBACK_KEEP),
        )
    except KeyboardInterrupt:
        print()
        print("Interrupted. No button was recorded.")
        return 130

    print()
    print("=" * 60)
    if choice == TELEGRAM_CALLBACK_PROMOTE:
        print("HEARD: Promote to best_model")
        print("  callback_data =", choice)
        print("  This is what finetune_latest treats as Accept.")
        print("  This test did NOT copy best_model.zip.")
        print("=" * 60)
        return 0
    if choice == TELEGRAM_CALLBACK_KEEP:
        print("HEARD: Keep current")
        print("  callback_data =", choice)
        print("  This is what finetune_latest treats as reject / stay.")
        print("=" * 60)
        return 0

    print("TIMEOUT: no Promote/Keep tap reached this process.")
    print("  Possible causes:")
    print("  - another script is also calling getUpdates (only one listener works)")
    print("  - you tapped after the wait window")
    print("  - the bot never received the callback (tap the buttons on THIS test message)")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
