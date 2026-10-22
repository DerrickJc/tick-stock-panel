"""盘后策略缓存预热 — 判重语义、提交链路与开关行为。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from types import SimpleNamespace

import pytest

from app.config import settings
from app.services import strategy_cache, strategy_prewarm, strategy_run_queue

AS_OF = "2026-10-12"


# ── Fakes (与 test_screener_run_all_progressive 同款风格) ──

@dataclass
class _FakeResult:
    total: int = 1
    rows: list = field(default_factory=lambda: [{"symbol": "000001.SZ", "close": 1.0}])
    as_of: str = AS_OF


class _FakeEngine:
    def __init__(self, ids: list[str]):
        self._ids = ids
        self.executed: list[str] = []

    def list_strategies(self) -> list[dict]:
        return [{"id": sid} for sid in self._ids]

    def has(self, sid: str) -> bool:
        return sid in self._ids

    def get(self, sid: str):
        return SimpleNamespace(meta={})

    def run_all(self, context, params_map=None, overrides_map=None, *, strategy_ids=None, parallel=True):
        out = {}
        for sid in strategy_ids or []:
            self.executed.append(sid)
            out[sid] = _FakeResult()
        return out


class _FakeService:
    def __init__(self, repo, asset_type="stock"):
        pass

    def latest_date(self):
        return date.fromisoformat(AS_OF)

    def build_strategy_context(self, *args, **kwargs):
        return SimpleNamespace()


def _app(tmp_path, engine):
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    state = SimpleNamespace(repo=repo, strategy_engine=engine, monitor_engine=None)
    return SimpleNamespace(state=state)


def _write_full_cache(tmp_path, ids, as_of=AS_OF):
    strategy_cache.write_cache(
        tmp_path, as_of,
        {sid: {"total": 1, "as_of": as_of, "rows": []} for sid in ids},
    )


def _wait_executed(engine, want, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if set(want).issubset(set(engine.executed)):
            return True
        time.sleep(0.05)
    return False


# ── needs_prewarm 判重语义 (与前端 cacheCoversPool 同口径) ──

def test_needs_prewarm_true_on_cold_cache(tmp_path):
    engine = _FakeEngine(["a", "b"])
    assert strategy_prewarm.needs_prewarm(tmp_path, engine, date.fromisoformat(AS_OF)) is True


def test_needs_prewarm_false_when_fully_covered(tmp_path):
    engine = _FakeEngine(["a", "b"])
    _write_full_cache(tmp_path, ["a", "b"])
    assert strategy_prewarm.needs_prewarm(tmp_path, engine, date.fromisoformat(AS_OF)) is False


def test_needs_prewarm_true_on_partial_coverage(tmp_path):
    engine = _FakeEngine(["a", "b", "c"])
    _write_full_cache(tmp_path, ["a", "b"])
    assert strategy_prewarm.needs_prewarm(tmp_path, engine, date.fromisoformat(AS_OF)) is True


def test_needs_prewarm_true_when_cache_date_behind(tmp_path):
    engine = _FakeEngine(["a"])
    _write_full_cache(tmp_path, ["a"], as_of="2026-10-10")
    assert strategy_prewarm.needs_prewarm(tmp_path, engine, date.fromisoformat(AS_OF)) is True


def test_needs_prewarm_true_when_entry_as_of_mismatch(tmp_path):
    # 顶层 as_of 一致但个别策略结果日期落后 (历史脏数据形态) → 该策略视为未覆盖
    engine = _FakeEngine(["a", "b"])
    strategy_cache.write_cache(
        tmp_path, AS_OF,
        {"a": {"total": 1, "as_of": AS_OF, "rows": []},
         "b": {"total": 1, "as_of": "2026-10-10", "rows": []}},
    )
    assert strategy_prewarm.needs_prewarm(tmp_path, engine, date.fromisoformat(AS_OF)) is True


def test_needs_prewarm_false_without_data_or_strategies(tmp_path):
    engine = _FakeEngine(["a"])
    assert strategy_prewarm.needs_prewarm(tmp_path, engine, None) is False
    empty = _FakeEngine([])
    assert strategy_prewarm.needs_prewarm(tmp_path, empty, date.fromisoformat(AS_OF)) is False


def test_eligible_daily_ids_filters_research_and_mismatched_axes():
    engine = SimpleNamespace(list_strategies=lambda: [
        {"id": "ok"},
        {"id": "research", "research_only": True},
        {"id": "etf_only", "asset_types": ["etf"]},
        {"id": "minute_only", "timeframes": ["1m"]},
        {"id": "default_axes"},
    ])
    assert strategy_prewarm.eligible_daily_ids(engine) == ["ok", "default_axes"]


# ── schedule_prewarm 提交链路 ──

def test_schedule_prewarm_submits_and_fills_cache(monkeypatch, tmp_path):
    from app.services import screener as screener_service
    monkeypatch.setattr(screener_service, "ScreenerService", _FakeService)

    engine = _FakeEngine(["a", "b"])
    app = _app(tmp_path, engine)
    strategy_prewarm.schedule_prewarm(app)

    assert _wait_executed(engine, {"a", "b"})
    results = (strategy_cache.read_cache(tmp_path) or {}).get("results") or {}
    assert set(results) == {"a", "b"}
    assert all(r.get("computed_at") for r in results.values())
    # 耗时记录落盘 → 下次按耗时排序的既有机制同样适用于预热
    assert set(strategy_run_queue.load_run_timings(tmp_path)) == {"a", "b"}


def test_schedule_prewarm_skips_when_covered(monkeypatch, tmp_path):
    from app.services import screener as screener_service
    monkeypatch.setattr(screener_service, "ScreenerService", _FakeService)

    engine = _FakeEngine(["a"])
    _write_full_cache(tmp_path, ["a"])
    strategy_prewarm.schedule_prewarm(_app(tmp_path, engine))

    assert engine.executed == []
    assert not (tmp_path / "user_data" / "strategy_run_timings.json").exists()


def test_schedule_prewarm_disabled_by_config(monkeypatch, tmp_path):
    from app.services import screener as screener_service
    monkeypatch.setattr(screener_service, "ScreenerService", _FakeService)
    monkeypatch.setattr(settings, "strategy_prewarm_enabled", False)

    engine = _FakeEngine(["a"])
    strategy_prewarm.schedule_prewarm(_app(tmp_path, engine))
    assert engine.executed == []

    # 脏状态隔离: 恢复开关后同一实例可正常提交
    monkeypatch.setattr(settings, "strategy_prewarm_enabled", True)
    strategy_prewarm.schedule_prewarm(_app(tmp_path, engine))
    assert _wait_executed(engine, {"a"})


def test_schedule_prewarm_swallows_errors(monkeypatch, tmp_path):
    # 预热绝不反噬数据管道: 服务构造抛错也只静默返回
    from app.services import screener as screener_service

    def _boom(repo):
        raise RuntimeError("repo unavailable")

    monkeypatch.setattr(screener_service, "ScreenerService", _boom)
    strategy_prewarm.schedule_prewarm(_app(tmp_path, _FakeEngine(["a"])))


def test_prewarm_rides_same_execution_as_page_run(monkeypatch, tmp_path):
    """预热与页面 run_all 同 key → 搭车同一执行, 不重复算。"""
    from app.services import screener as screener_service
    monkeypatch.setattr(screener_service, "ScreenerService", _FakeService)

    engine = _FakeEngine(["a"])
    app = _app(tmp_path, engine)
    strategy_prewarm.schedule_prewarm(app)
    # 页面在预热运行中发起同参数 run_all (等价 key) → 搭车
    resp = strategy_prewarm.run_all_progressive(
        repo=app.state.repo, engine=engine, svc=_FakeService(app.state.repo),
        as_of=date.fromisoformat(AS_OF), asset_type="stock", timeframe="1d",
        all_ids=["a"], params_map={}, overrides_map={}, first_return_s=3.0,
    )
    assert resp["complete"] is True or resp["results"].get("a", {}).get("total") == 1
    # 只执行一次
    assert engine.executed.count("a") == 1
