"""Test lecture HR Garmin — utilise la meme session BLE que webcam_rppg_live.py."""

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.garmin_ble_reader import run_garmin_hr_session


async def main() -> None:
    print("Recherche directe de la Garmin Forerunner...")
    print("La montre doit rester en mode : Diffuser la frequence cardiaque.\n")

    def on_hr(hr: int) -> None:
        print(f"HR Garmin : {hr} bpm")

    def on_status(msg: str) -> None:
        print(msg)

    await run_garmin_hr_session(
        scan_timeout=20.0,
        connect_timeout=30.0,
        on_hr=on_hr,
        on_status=on_status,
    )


if __name__ == "__main__":
    asyncio.run(main())
