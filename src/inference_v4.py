"""V4 paper inference — 1082-dim Bloomberg observers, volume=0.

Does not write state.pkl / state_v3.pkl / enhanced_v3.parquet.
Own book: ``state_v4.pkl``. Own panel: ``enhanced_v4.parquet``.

Do not point this at a V2 zip (782) or a V3 zip (TradingView volume).
Do not run this while V3.2 is trading the same SIMULATE account.

    python -m src.inference_v4 --predict-now --model models/news_gpu_v4/best_model.zip
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader_v4 import ENHANCED_V4_PARQUET
from src.futu_codes import FUTU_KLINE_ALIASES_V4
from src.trading_env_v4 import TradingEnv, observation_dim
from src.utils import PROJECT_ROOT, setup_logging
import src.inference_v3 as v3

logger = setup_logging("airaire.inference_v4")

STATE_V4_PKL = PROJECT_ROOT / "state_v4.pkl"
V4_OBS_DIM = observation_dim()


def _refuse_wrong_family(path: Path) -> None:
    text = str(Path(path).resolve()).replace("\\", "/")
    if "news_gpu_v2" in text:
        raise ValueError(f"inference_v4 refuses {path} (V2 782-dim).")
    if "news_gpu_v3" in text:
        raise ValueError(
            f"inference_v4 refuses {path} (V3 learned TradingView volume). Train/load a news_gpu_v4 zip."
        )


def make_v4_env(df, news_scores, *, long_only: bool):
    if long_only:
        logger.warning("V4 is not long-only. Ignoring long_only for the env (still 5-name [-1,1] action).")
    return TradingEnv(df=df, news_scores=news_scores)


def _load_v4_policy(model_path: Path):
    _refuse_wrong_family(model_path)
    return v3.load_v3_policy(model_path)


def _resolve_v4_model(explicit: Path | None = None) -> Path:
    if explicit is not None:
        _refuse_wrong_family(explicit)
        return Path(explicit)
    cand = Path("models") / "news_gpu_v4" / "best_model.zip"
    return cand


def _patch() -> None:
    v3.ENHANCED_V3_PARQUET = ENHANCED_V4_PARQUET
    v3.STATE_V3_PKL = STATE_V4_PKL
    v3.V3_OBS_DIM = V4_OBS_DIM
    v3.FUTU_KLINE_ALIASES = FUTU_KLINE_ALIASES_V4
    v3.make_v3_env = make_v4_env
    v3.load_v3_policy = _load_v4_policy
    v3.resolve_v3_model_path = _resolve_v4_model
    orig_refuse = v3._refuse_v2_zip

    def _refuse(path: Path) -> None:
        orig_refuse(path)
        _refuse_wrong_family(path)

    v3._refuse_v2_zip = _refuse


def main() -> None:
    import signal

    _patch()
    logger.info("AirAire inference V4 — panel=%s state=%s obs_dim=%d", ENHANCED_V4_PARQUET, STATE_V4_PKL, V4_OBS_DIM)
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
        long_only_flag=args.long_only,
        reduce_only=not args.allow_shorts,
        push_dashboard=args.push_dashboard,
    )


if __name__ == "__main__":
    main()
