"""V2.10 paper inference — 932-dim Bloomberg lookback (5 core + HSI), long-only.

Does not write state.pkl / state_v3.pkl / enhanced_data.parquet.
Own book: ``state_v2_10.pkl``. Own panel: ``enhanced_v2_10.parquet``.

Do not point this at a V2 zip (782) or a V3/V4 zip (1082, SPX).
Do not run this while live V2 is trading the same SIMULATE account.

    python -m src.inference_v2_10 --predict-now --model models/news_gpu_v2_10/best_model.zip
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader_v2_10 import ENHANCED_V2_10_PARQUET, _zero_volume
from src.futu_codes import FUTU_KLINE_ALIASES_V2_10
from src.trading_env_v2_10 import OBS_TICKERS, OBSERVER_TICKERS_V2_10, TradingEnv, observation_dim
from src.utils import PROJECT_ROOT, setup_logging
import src.inference_v3 as v3

logger = setup_logging("airaire.inference_v2_10")

STATE_V2_10_PKL = PROJECT_ROOT / "state_v2_10.pkl"
V2_10_OBS_DIM = observation_dim()


def _refuse_wrong_family(path: Path) -> None:
    parts = {str(p) for p in Path(path).resolve().parts}
    if "news_gpu_v2_10" in parts:
        return
    if "news_gpu_v2" in parts:
        raise ValueError(f"inference_v2_10 refuses {path} (V2 782-dim).")
    if "news_gpu_v2_11" in parts:
        raise ValueError(f"inference_v2_10 refuses {path} (V2.11 782-dim hybrid).")
    blocked = {
        "news_gpu_v3",
        "news_gpu_v3_1",
        "news_gpu_v3_2",
        "news_gpu_v4",
        "news_gpu_v4_1",
        "news",
        "news_gpu",
    }
    if parts & blocked:
        raise ValueError(f"inference_v2_10 refuses {path} (wrong family / SPX / shorts). Train news_gpu_v2_10.")


def make_v2_10_env(df, news_scores, *, long_only: bool):
    if not long_only:
        logger.warning("V2.10 is long-only. Ignoring long_only=False.")
    return TradingEnv(df=df, news_scores=news_scores)


_ORIG_LOAD = None


def _load_v2_10_policy(model_path: Path):
    _refuse_wrong_family(model_path)
    return _ORIG_LOAD(model_path)


def _resolve_v2_10_model(explicit: Path | None = None) -> Path:
    if explicit is not None:
        _refuse_wrong_family(explicit)
        return Path(explicit)
    return Path("models") / "news_gpu_v2_10" / "best_model.zip"


def _is_long_only(_model_path, _flag, _model=None) -> bool:
    return True


def _patch() -> None:
    from src.data_loader import overlay_live_ohlcv

    global _ORIG_LOAD
    if _ORIG_LOAD is None:
        _ORIG_LOAD = v3.load_v3_policy

    v3.ENHANCED_V3_PARQUET = ENHANCED_V2_10_PARQUET
    v3.STATE_V3_PKL = STATE_V2_10_PKL
    v3.V3_OBS_DIM = V2_10_OBS_DIM
    v3.FUTU_KLINE_ALIASES = FUTU_KLINE_ALIASES_V2_10
    v3.ALL_TICKERS = list(OBS_TICKERS)
    v3.OBSERVER_TICKERS = list(OBSERVER_TICKERS_V2_10)
    v3.make_v3_env = make_v2_10_env
    v3.load_v3_policy = _load_v2_10_policy
    v3.resolve_v3_model_path = _resolve_v2_10_model
    v3._is_long_only = _is_long_only
    v3.overlay_live_ohlcv = lambda panel, live, now=None: _zero_volume(overlay_live_ohlcv(panel, live, now=now))
    orig_refuse = v3._refuse_v2_zip

    def _refuse(path: Path) -> None:
        orig_refuse(path)
        _refuse_wrong_family(path)

    v3._refuse_v2_zip = _refuse


def main() -> None:
    import signal

    _patch()
    logger.info(
        "AirAire inference V2.10 — panel=%s state=%s obs_dim=%d long-only HSI-only",
        ENHANCED_V2_10_PARQUET,
        STATE_V2_10_PKL,
        V2_10_OBS_DIM,
    )
    args = v3.parse_args()
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, v3._handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, v3._handle_stop)
    v3.run_loop(
        once=args.once,
        dry_run=args.dry_run,
        poll_seconds=args.poll_seconds,
        model_path=args.model,
        skip_catch_up=args.skip_catch_up,
        predict_now=args.predict_now,
        long_only_flag=True,
        reduce_only=True,
        push_dashboard=args.push_dashboard,
    )


if __name__ == "__main__":
    main()
