"""
Lecture HR Garmin Forerunner en Bluetooth LE.

Le script tools/test_garmin_ble_hr.py utilise run_garmin_hr_session() ci-dessous.
Pour la webcam, GarminBLEReader lance la meme logique dans un PROCESSUS separe
(asyncio.run comme le test) pour eviter les conflits avec OpenCV.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import struct
import time
from typing import Callable, List, Optional, Tuple

HEART_RATE_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_SERVICE_SHORT = "180d"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


def parse_heart_rate(data: bytearray) -> int:
    flags = data[0]
    hr_is_uint16 = flags & 0x01
    if hr_is_uint16:
        return int(struct.unpack_from("<H", data, 1)[0])
    return int(data[1])


def is_garmin_filter(device, advertisement_data) -> bool:
    """Filtre identique a test_garmin_ble_hr.py."""
    name = device.name or advertisement_data.local_name or ""
    return "forerunner" in name.lower() or "garmin" in name.lower()


async def run_garmin_hr_session(
    scan_timeout: float = 20.0,
    connect_timeout: float = 30.0,
    device_address: Optional[str] = None,
    on_hr: Optional[Callable[[int], None]] = None,
    on_status: Optional[Callable[[str], None]] = None,
    on_connected: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> bool:
    """
    Session BLE = copie de test_garmin_ble_hr.py (find -> connect -> notify).

    Returns:
        True si connecte puis arrete proprement, False si montre introuvable / erreur.
    """
    from bleak import BleakClient, BleakScanner

    def status(msg: str) -> None:
        if on_status:
            on_status(msg)

    status("Recherche directe de la Garmin Forerunner...")
    status("La montre doit rester en mode : Diffuser la frequence cardiaque.")

    if device_address:
        device = await BleakScanner.find_device_by_address(
            device_address,
            timeout=scan_timeout,
        )
    else:
        device = await BleakScanner.find_device_by_filter(
            is_garmin_filter,
            timeout=scan_timeout,
        )

    if device is None:
        status("Aucune Garmin trouvee.")
        status("Verifiez mode diffuser FC et BLE telephone OFF.")
        return False

    name = device.name or device.address
    status(f"Garmin trouvee : {name} | {device.address}")
    status("Connexion en cours...")

    try:
        async with BleakClient(device, timeout=connect_timeout) as client:
            status("Connecte.")

            try:
                services = await client.get_services()
            except Exception:
                services = client.services

            service_uuids = [service.uuid.lower() for service in services]
            if HEART_RATE_SERVICE_UUID not in service_uuids:
                status("Service Heart Rate standard non trouve.")
                return False

            status("Service Heart Rate trouve. Lecture FC...")

            def callback(sender, data: bytearray) -> None:
                hr = parse_heart_rate(data)
                if on_hr:
                    on_hr(hr)

            await client.start_notify(HEART_RATE_MEASUREMENT_UUID, callback)

            if on_connected:
                on_connected(name)
            status(f"Garmin OK : {name}")

            while True:
                if should_stop and should_stop():
                    break
                await asyncio.sleep(0.25)

            return True

    except Exception as exc:
        status(f"Erreur connexion Bluetooth : {exc}")
        return False


# ---------------------------------------------------------------------------
# Processus worker (meme asyncio.run que test_garmin_ble_hr.py)
# ---------------------------------------------------------------------------


def _ble_process_main(
    hr_queue: mp.Queue,
    status_queue: mp.Queue,
    connected_flag: mp.Value,
    device_name_buf: mp.Array,
    stop_flag: mp.Value,
    scan_timeout: float,
    connect_timeout: float,
    device_address: str,
) -> None:
    """Point d'entree du processus fils (spawn Windows)."""

    def on_hr(hr: int) -> None:
        try:
            hr_queue.put_nowait(float(hr))
        except Exception:
            pass

    def on_status(msg: str) -> None:
        try:
            status_queue.put_nowait(msg)
        except Exception:
            pass

    def on_connected(name: str) -> None:
        connected_flag.value = True
        encoded = name.encode("utf-8", errors="replace")[:255]
        for i, byte in enumerate(encoded):
            device_name_buf[i] = byte
        device_name_buf[len(encoded)] = b"\0"[0]

    def should_stop() -> bool:
        return bool(stop_flag.value)

    async def outer_loop() -> None:
        addr = device_address.strip() or None
        while not should_stop():
            await run_garmin_hr_session(
                scan_timeout=scan_timeout,
                connect_timeout=connect_timeout,
                device_address=addr,
                on_hr=on_hr,
                on_status=on_status,
                on_connected=on_connected,
                should_stop=should_stop,
            )
            connected_flag.value = False
            if should_stop():
                break
            on_status("Reconnexion dans 4 s...")
            await asyncio.sleep(4)

    asyncio.run(outer_loop())


# ---------------------------------------------------------------------------
# API pour webcam_rppg_live.py
# ---------------------------------------------------------------------------


class GarminBLEReader:
    """Lit la HR Garmin dans un processus dedie (comme le script de test seul)."""

    def __init__(
        self,
        scan_timeout: float = 20.0,
        connect_timeout: float = 30.0,
        stale_seconds: float = 5.0,
        device_address: Optional[str] = None,
        debug: bool = False,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.scan_timeout = scan_timeout
        self.connect_timeout = connect_timeout
        self.stale_seconds = stale_seconds
        self.device_address = (device_address or "").strip()
        self.debug = debug
        self._on_status = on_status

        self._hr: Optional[float] = None
        self._hr_timestamp = 0.0
        self._status = "Demarrage processus Garmin..."
        self._ctx = mp.get_context("spawn")
        self._hr_queue: Optional[mp.Queue] = None
        self._status_queue: Optional[mp.Queue] = None
        self._connected_flag: Optional[mp.Value] = None
        self._device_name_buf: Optional[mp.Array] = None
        self._stop_flag: Optional[mp.Value] = None
        self._process: Optional[mp.Process] = None

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return

        self._hr_queue = self._ctx.Queue()
        self._status_queue = self._ctx.Queue()
        self._connected_flag = self._ctx.Value("b", False)
        self._device_name_buf = self._ctx.Array("c", 256)
        self._stop_flag = self._ctx.Value("b", False)

        self._process = self._ctx.Process(
            target=_ble_process_main,
            args=(
                self._hr_queue,
                self._status_queue,
                self._connected_flag,
                self._device_name_buf,
                self._stop_flag,
                float(self.scan_timeout),
                float(self.connect_timeout),
                self.device_address,
            ),
            name="GarminBLE",
            daemon=True,
        )
        self._process.start()
        self._status = "Processus Garmin demarre (asyncio.run)..."

    def stop(self) -> None:
        if self._stop_flag is not None:
            self._stop_flag.value = True
        if self._process is not None:
            self._process.join(timeout=self.scan_timeout + 15.0)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None
        self._hr = None
        self._status = "Garmin BLE arrete."

    @property
    def is_connected(self) -> bool:
        self._poll_queues()
        return bool(self._connected_flag and self._connected_flag.value)

    @property
    def device_name(self) -> str:
        if self._device_name_buf is None:
            return ""
        raw = bytes(self._device_name_buf[:]).split(b"\0", 1)[0]
        return raw.decode("utf-8", errors="replace")

    def get_status(self) -> str:
        self._poll_queues()
        return self._status

    def get_hr(self) -> Optional[float]:
        self._poll_queues()
        if self._hr is None:
            return None
        if time.time() - self._hr_timestamp > self.stale_seconds:
            return None
        return self._hr

    def _poll_queues(self) -> None:
        if self._status_queue is not None:
            while True:
                try:
                    msg = self._status_queue.get_nowait()
                except Exception:
                    break
                self._status = msg
                if self._on_status:
                    self._on_status(msg)

        if self._hr_queue is not None:
            while True:
                try:
                    hr = self._hr_queue.get_nowait()
                except Exception:
                    break
                self._hr = hr
                self._hr_timestamp = time.time()


# ---------------------------------------------------------------------------
# Scan diagnostic
# ---------------------------------------------------------------------------


def _device_display_name(device, advertisement_data=None) -> str:
    if device.name:
        return device.name
    if advertisement_data is not None and advertisement_data.local_name:
        return advertisement_data.local_name
    return device.address or "?"


def _adv_has_heart_rate_service(advertisement_data) -> bool:
    if advertisement_data is None or not advertisement_data.service_uuids:
        return False
    for uuid in advertisement_data.service_uuids:
        if HEART_RATE_SERVICE_SHORT in uuid.lower().replace("-", ""):
            return True
    return False


async def scan_ble_devices(timeout: float = 12.0) -> List[Tuple[str, str, bool, bool]]:
    from bleak import BleakScanner

    rows: List[Tuple[str, str, bool, bool]] = []
    try:
        raw = await BleakScanner.discover(timeout=timeout, return_adv=True)
    except TypeError:
        for device in await BleakScanner.discover(timeout=timeout):
            rows.append((device.address, _device_display_name(device), False, False))
        return rows

    items = raw.values() if isinstance(raw, dict) else raw
    for item in items:
        if isinstance(item, tuple) and len(item) >= 2:
            device, adv = item[0], item[1]
        else:
            device, adv = item, None
        name = _device_display_name(device, adv)
        lowered = name.lower()
        rows.append(
            (
                device.address,
                name,
                "garmin" in lowered or "forerunner" in lowered,
                _adv_has_heart_rate_service(adv),
            )
        )
    return rows


def print_ble_scan_report(timeout: float = 15.0) -> int:
    async def _run() -> int:
        print(f"Scan BLE ({timeout:.0f}s) - mode Diffuser la FC\n")
        rows = await scan_ble_devices(timeout=timeout)
        if not rows:
            print("Aucun peripherique BLE detecte.")
            return 1
        print(f"{'ADRESSE':<20} {'NOM':<28} {'GARMIN?':<8} {'HR SVC?'}")
        print("-" * 70)
        for address, name, is_garmin, has_hr in rows:
            print(
                f"{address:<20} {name[:28]:<28} "
                f"{'oui' if is_garmin else '':<8} {'oui' if has_hr else ''}"
            )
        print(
            "\n  python tools/webcam_rppg_live.py --method pos --garmin "
            "--garmin-address ADRESSE"
        )
        return 0

    return asyncio.run(_run())
