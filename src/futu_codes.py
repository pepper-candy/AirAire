"""OpenD quote codes for observers. Shared by the Futu probe and V4 overlay.

HSI: HK.800000 already worked on 24 Aug replay.
Cash SPX: US.SPX and US..SPX failed that same run (\"US stock indices are not supported\").
SPY/VOO/IVV are ETFs — useful as a live proxy only if we also train on that same code.
Do not train on Bloomberg SPX and then feed SPY into the SPX slot.
"""

from __future__ import annotations

HSI_CODES: tuple[str, ...] = ("HK.800000", "HK.HSI")
SPX_INDEX_CODES: tuple[str, ...] = ("US.SPX", "US..SPX")
SPX_PROXY_ETFS: tuple[str, ...] = ("US.SPY", "US.VOO", "US.IVV")

# History/snapshot try-list. Probe prints which ones actually return bars.
OBSERVER_PROBE_CODES: tuple[str, ...] = HSI_CODES + SPX_INDEX_CODES + SPX_PROXY_ETFS

FUTU_KLINE_ALIASES_V4: dict[str, tuple[str, ...]] = {
    "HK.00700": ("HK.00700",),
    "HK.03690": ("HK.03690",),
    "HK.03750": ("HK.03750",),
    "US.COST": ("US.COST",),
    "US.KO": ("US.KO",),
    "HK.HSI": HSI_CODES,
    # Index codes first. ETFs are last so a real SPX feed wins if OpenD ever adds it.
    "US.SPX": SPX_INDEX_CODES + SPX_PROXY_ETFS,
}
