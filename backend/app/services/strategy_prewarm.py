"""盘后策略缓存预热 — 数据落地后主动算满当日策略缓存, 策略页打开即读缓存。

设计要点:
- 与策略页 run_all 走同一条渐进式执行编排 (本模块的 run_all_progressive,
  从 api/screener 提取): 同 key 单飞, 预热运行中用户打开页面会搭车同一执行,
  不重复算; 逐策略算完即写 strategy_cache, 前端轮询可逐个点亮。
- 触发点链在 repo._on_refresh_done 上 (定时盘后管道 / 数据页手动同步 /
  启动预热三个完成点都会触发); needs_prewarm 按「缓存 as_of 与最新 enriched
  日期一致 且 results 覆盖全部合格策略」判重, 已暖则跳过。
"""
from __future__ import annotations

import contextlib
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import date
from typing import Any

from app.config import settings
from app.services import strategy_cache, strategy_run_queue

logger = logging.getLogger(__name__)


# ── 轻量助手 (与 api/screener._safe 同语义; services 层不反向依赖 api 模块) ──

def _safe_rows(result: Any) -> list[dict]:
    rows = asdict(result).get("rows", [])
    for r in rows:
        for k, v in list(r.items()):
            if isinstance(v, float) and not math.isfinite(v):
                r[k] = None
    return rows


def coverage_warnings(
    svc, as_of, *, engine=None, strategy_ids=None, params_map=None, overrides_map=None,
) -> list[str]:
    """数据充足性提示 (#303): enriched 覆盖低于暖机需求时给出人话警告。

    advisory 元数据: 任何计算失败都静默返回 [], 绝不影响策略主流程。
    引擎/服务无该方法时 (测试 Fake) 同样跳过, 与 build_shared_matrix 的
    可选能力探测同风格。
    """
    try:
        required: int | None = None
        rhb = getattr(engine, "required_history_bars", None)
        if engine is not None and strategy_ids and callable(rhb):
            required = rhb(strategy_ids, params_map=params_map, overrides_map=overrides_map)
        cw = getattr(svc, "coverage_warnings", None)
        if callable(cw):
            return cw(as_of, required_bars=required)
    except Exception:
        logger.debug("coverage warnings unavailable", exc_info=True)
    return []


def eligible_daily_ids(engine, asset_type: str = "stock", timeframe: str = "1d") -> list[str]:
    """与 run_all 默认路径同口径的合格策略清单 (非 research_only / 匹配资产与周期)。"""
    return [
        meta["id"]
        for meta in engine.list_strategies()
        if not meta.get("research_only")
        and asset_type in meta.get("asset_types", ["stock"])
        and timeframe in meta.get("timeframes", ["1d"])
    ]


# ── 渐进式执行编排 (api/screener._run_all_progressive 提取, 端点与预热共用) ──

def _build_job(
    *, repo, engine, svc, as_of: date, timeframe: str,
    all_ids: list[str], params_map: dict, overrides_map: dict,
) -> Callable[[strategy_run_queue.StrategyRunHandle], None]:
    data_dir = repo.store.data_dir

    def job(handle: strategy_run_queue.StrategyRunHandle) -> None:
        context = svc.build_strategy_context(
            engine,
            as_of,
            all_ids,
            timeframe=timeframe,
            params_map=params_map,
            overrides_map=overrides_map,
        )
        # 逐策略 run_all 不会把矩阵回写 context.market → 每个矩阵策略都会重建
        # 全市场矩阵 (小服务器上单次数秒到十余秒)。这里按字段并集一次建好复用;
        # FakeEngine 等无该方法的实现跳过 (保持旧行为)。
        if getattr(context, "market", None) is None:
            build_matrix = getattr(engine, "build_shared_matrix", None)
            if callable(build_matrix):
                matrix = build_matrix(
                    context,
                    [(sid, engine.get(sid)) for sid in all_ids],
                    params_map,
                    overrides_map,
                )
                if matrix is not None:
                    context = replace(context, market=matrix)
        all_results: dict[str, dict] = {}
        elapsed_map: dict[str, float] = {}
        for sid in all_ids:
            t0 = time.perf_counter()
            # 逐策略隔离: 单个策略崩溃 (如自定义代码的数据类型错误) 只记
            # 错误跳过, 不让整批剩余策略陪葬 — 其余策略照常算完落缓存。
            try:
                single = engine.run_all(
                    context,
                    params_map=params_map,
                    overrides_map=overrides_map,
                    strategy_ids=[sid],
                    parallel=False,
                )
                result = single[sid]
            except Exception as e:  # noqa: BLE001
                logger.warning("run_all: 策略 %s 执行失败, 跳过: %s", sid, e, exc_info=True)
                handle.fail_one(sid, str(e))
                continue
            payload = {
                "total": result.total,
                "as_of": str(as_of),
                "rows": _safe_rows(result),
                "computed_at": int(time.time() * 1000),
            }
            if timeframe == "1d":
                w = coverage_warnings(
                    svc, as_of, engine=engine, strategy_ids=[sid],
                    params_map=params_map, overrides_map=overrides_map,
                )
                if w:
                    payload["warnings"] = w
            all_results[sid] = payload
            elapsed_map[sid] = (time.perf_counter() - t0) * 1000
            # 逐策略增量落盘 (write_cache 同日按 sid 合并), 前端轮询即可逐个看到
            try:
                strategy_cache.write_cache(data_dir, str(as_of), {sid: payload})
            except Exception:
                logger.warning("run_all 渐进写入缓存失败: %s", sid, exc_info=True)
            handle.complete(sid, {k: v for k, v in payload.items() if k != "rows"})
        # 收尾: 与旧版口径一致的整体重写 + 耗时落盘供下次排序
        if all_results:
            with contextlib.suppress(Exception):
                strategy_cache.write_cache(data_dir, str(as_of), all_results)
        strategy_run_queue.record_run_timings(data_dir, elapsed_map)

    return job


def submit_progressive(
    *, repo, engine, svc, as_of: date, asset_type: str, timeframe: str,
    all_ids: list[str], params_map: dict, overrides_map: dict,
) -> strategy_run_queue.StrategyRunHandle:
    """组装 key 与耗时排序, 提交到单飞执行器; 不等待 (fire-and-forget)。"""
    data_dir = repo.store.data_dir
    key = (asset_type, timeframe, str(as_of), tuple(sorted(all_ids)))
    ordered_ids = strategy_run_queue.order_strategy_ids(
        all_ids, strategy_run_queue.load_run_timings(data_dir)
    )
    job = _build_job(
        repo=repo, engine=engine, svc=svc, as_of=as_of, timeframe=timeframe,
        all_ids=ordered_ids, params_map=params_map, overrides_map=overrides_map,
    )
    return strategy_run_queue.MANAGER.get_or_submit(key, ordered_ids, job)


def run_all_progressive(
    *,
    repo,
    engine,
    svc,
    as_of: date,
    asset_type: str,
    timeframe: str,
    all_ids: list[str],
    params_map: dict,
    overrides_map: dict,
    first_return_s: float,
    t_total: float | None = None,
) -> dict:
    """run_all 渐进式执行: 快策略随响应先返回, 慢策略后台算完逐个落缓存。

    执行全程在单飞执行器里 (见 services/strategy_run_queue.py): 相同请求
    搭车现有执行, 不同请求排队; HTTP 侧只轮询状态快照到首返时限。
    """
    if t_total is None:
        t_total = time.perf_counter()
    handle = submit_progressive(
        repo=repo, engine=engine, svc=svc, as_of=as_of, asset_type=asset_type,
        timeframe=timeframe, all_ids=all_ids, params_map=params_map,
        overrides_map=overrides_map,
    )
    deadline = time.perf_counter() + first_return_s
    snap = handle.snapshot()
    while not snap["done"] and time.perf_counter() < deadline:
        time.sleep(0.2)
        snap = handle.snapshot()

    done_results = snap["results"]
    if snap["error"] and not done_results:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=snap["error"])
    logger.info(
        "run_all: first return %.1fms (%d done, %d pending)",
        (time.perf_counter() - t_total) * 1000,
        len(done_results),
        len(snap["pending"]),
    )
    return {
        "as_of": str(as_of),
        "results": done_results,
        "pending": snap["pending"],
        "errors": snap["errors"],
        "complete": snap["done"] and not snap["errors"],
        "error": snap["error"],
        "started_at": snap["started_at_ms"],
    }


# ── 盘后预热入口 ──────────────────────────────────────────────────────

def needs_prewarm(
    data_dir, engine, latest_date: date | None, *, asset_type: str = "stock",
) -> bool:
    """当日策略缓存是否需要预热。

    与前端 cacheCoversPool 同语义: 缓存 as_of 等于最新 enriched 日期, 且
    每个合格策略都有当日结果才算已暖; 无数据 / 无策略时 False (无事可做)。
    """
    if latest_date is None:
        return False
    ids = eligible_daily_ids(engine, asset_type)
    if not ids:
        return False
    cached = strategy_cache.read_cache(data_dir)
    as_of_s = str(latest_date)
    if not cached or str(cached.get("as_of") or "") != as_of_s:
        return True
    results = cached.get("results") or {}
    covered = {
        sid for sid in ids
        if isinstance(results.get(sid), dict) and results[sid].get("as_of") == as_of_s
    }
    return not set(ids).issubset(covered)


def schedule_prewarm(app) -> None:
    """数据落地后预热当日策略缓存; 由 repo._on_refresh_done 链式触发。

    判重 + 提交都在调用线程完成 (毫秒级: 读一个 JSON + 列策略), 实际计算在
    单飞执行器的 daemon 工作线程里跑, 与页面触发的 run_all 排同一条队列。
    任何失败只记日志, 绝不影响数据管道本身。
    """
    if not settings.strategy_prewarm_enabled:
        return
    try:
        from app.services import screener as screener_service
        from app.strategy import config as strategy_config

        repo = app.state.repo
        engine = getattr(app.state, "strategy_engine", None)
        if engine is None:
            return
        data_dir = repo.store.data_dir
        svc = screener_service.ScreenerService(repo)
        latest = svc.latest_date()
        if not needs_prewarm(data_dir, engine, latest):
            logger.debug("strategy prewarm skipped: cache already covers %s", latest)
            return
        all_ids = eligible_daily_ids(engine)
        if not all_ids:
            return
        overrides = strategy_config.list_overrides(data_dir)
        params_map = {
            sid: dict((overrides.get(sid) or {}).get("params") or {})
            for sid in all_ids
        }
        overrides_map = {sid: overrides.get(sid, {}) for sid in all_ids}
        submit_progressive(
            repo=repo, engine=engine, svc=svc, as_of=latest, asset_type="stock",
            timeframe="1d", all_ids=all_ids, params_map=params_map,
            overrides_map=overrides_map,
        )
        logger.info(
            "策略缓存预热已提交: %d 个策略, as_of=%s", len(all_ids), latest,
        )
    except Exception:  # noqa: BLE001
        logger.exception("策略缓存预热提交失败 (忽略)")
