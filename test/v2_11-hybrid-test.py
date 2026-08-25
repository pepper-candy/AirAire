"""V2.11 hybrid checks — no OpenD, no GPU required.

    python test/v2_11-hybrid-test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.order_lifecycle import (
    classify_place_error,
    decide_order,
    is_stale_working,
    skip_tiny_rebalance,
)
from src.trading_env import observation_dim
from src.trading_env_v2_11 import TradingEnv
from src.utils import CORE_TICKERS, NEWS_GPU_V2_11_MODELS_DIR, NEWS_GPU_V2_MODELS_DIR
from src.v2_11 import (
    FORBIDDEN_SEED_NAMES,
    clip_hybrid_action,
    guard_output_dir,
    refuse_wrong_inference_zip,
    resolve_v2_paper_seed_zip,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_obs_dim() -> None:
    _assert(observation_dim() == 782, f"V2 family obs_dim must be 782, got {observation_dim()}")
    env = TradingEnv()
    obs, _ = env.reset()
    _assert(int(obs.shape[0]) == 782, f"env obs {obs.shape}")
    _assert(float(env.action_space.low[0]) == -1.0, "Box low must stay -1 so V2 zips load")
    _assert(float(env.action_space.high[0]) == 1.0, "Box high must stay +1")


def test_hk_clip() -> None:
    raw = np.array([-0.9, -0.4, 0.2, -0.7, 0.5], dtype=np.float64)
    clipped = clip_hybrid_action(raw)
    _assert(clipped[0] == 0.0 and clipped[1] == 0.0, f"HK shorts must clip to 0: {clipped}")
    _assert(clipped[2] == 0.2, f"HK long kept: {clipped}")
    _assert(clipped[3] == -0.7 and clipped[4] == 0.5, f"US must stay free: {clipped}")
    env = TradingEnv()
    env.reset()
    _, _, _, _, info = env.step(raw)
    executed = info["action"]
    for ticker in CORE_TICKERS[:3]:
        _assert(executed[ticker] >= -1e-12, f"{ticker} executed {executed[ticker]} < 0")
    _assert(executed["US.COST"] < 0, f"US short should survive: {executed}")
    _assert(np.all(env._holdings[:3] >= -1e-12), f"HK holdings {env._holdings[:3]}")


def test_classify() -> None:
    _assert(classify_place_error("不支持卖空", ticker="HK.00700") == "hk_short", "卖空")
    _assert(classify_place_error("cannot short this stock", ticker="HK.03690") == "hk_short", "cannot short")
    _assert(classify_place_error("short selling is not allowed", ticker="HK.03750") == "hk_short", "short selling")
    _assert(classify_place_error("buying power insufficient", ticker="US.KO") == "buying_power", "power")
    _assert(classify_place_error("价格精度不正确", ticker="US.KO") == "price_precision", "tick")
    _assert(classify_place_error("手数不正确", ticker="HK.00700") == "lot_qty", "lot")
    _assert(classify_place_error("some new OpenD string xyz", ticker="US.COST") == "unknown", "unknown")
    # US shorts are legal — do not classify a US order as hk_short on a generic 'short' if we skipped US.
    _assert(classify_place_error("shortage of cash", ticker="US.COST") == "unknown", "shortage != short")


def test_min_notional() -> None:
    reason = skip_tiny_rebalance(
        ticker="US.COST",
        delta=7,
        px=900.0,
        current=0.0,
        equity=1_000_000.0,
        target_weight=0.006,
    )
    _assert(reason is not None, "7 COST shares must skip")
    reason14 = skip_tiny_rebalance(
        ticker="US.COST",
        delta=14,
        px=900.0,
        current=0.0,
        equity=1_000_000.0,
        target_weight=0.0126,
    )
    _assert(reason14 is None, f"14 COST shares should pass, got {reason14}")
    flatten = skip_tiny_rebalance(
        ticker="US.COST",
        delta=-7,
        px=900.0,
        current=7.0,
        equity=1_000_000.0,
        target_weight=0.0,
    )
    _assert(flatten is None, "flattening leftover 7 COST must be allowed")


def test_decide_order_roundtrip() -> None:
    skip = decide_order(
        ticker="US.KO",
        is_buy=False,
        qty=14,
        px=97.50,
        pending=[],
        last_buy_px=98.12,
        last_sell_px=None,
    )
    _assert(skip.action == "skip", f"must skip sell below last buy: {skip}")


def test_stale() -> None:
    row = {"bar_id": "old-bar", "ticker": "US.KO", "order_id": "1"}
    _assert(is_stale_working(row, current_bar_id="new-bar"), "previous bar is stale")
    fresh = {"bar_id": "new-bar", "ticker": "US.KO", "order_id": "1"}
    _assert(not is_stale_working(fresh, current_bar_id="new-bar"), "same bar is not stale")


def test_guards() -> None:
    _assert("checkpoint_2026-08-20.zip" in FORBIDDEN_SEED_NAMES, "must refuse guessed Aug-20 zip")
    try:
        guard_output_dir(NEWS_GPU_V2_MODELS_DIR)
        raise AssertionError("must refuse writing news_gpu_v2")
    except ValueError:
        pass
    out = guard_output_dir(NEWS_GPU_V2_11_MODELS_DIR)
    _assert(out == NEWS_GPU_V2_11_MODELS_DIR, "v2.11 dir ok")
    try:
        refuse_wrong_inference_zip(Path("models/old/news_gpu_v2_test/checkpoint_2026-08-20.zip"))
        raise AssertionError("must refuse 2026-08-20 filename")
    except ValueError:
        pass
    seed = resolve_v2_paper_seed_zip()
    _assert(seed.name == "best_model.zip", f"seed must be best_model.zip, got {seed}")
    _assert("2026-08-20" not in seed.name, seed)
    print(f"seed zip: {seed.resolve()}")


if __name__ == "__main__":
    tests = [
        test_obs_dim,
        test_hk_clip,
        test_classify,
        test_min_notional,
        test_decide_order_roundtrip,
        test_stale,
        test_guards,
    ]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print("V2.11 hybrid tests passed.")
