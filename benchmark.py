"""基准对比模块：获取基准指数数据，计算策略相对收益。

参考 vnpy BacktestingEngine 的基准对比设计：
  - 策略收益 vs 基准收益（alpha）
  - 策略波动 vs 基准波动
  - 策略最大回撤 vs 基准最大回撤

默认基准为沪深300指数（000300），可通过 config.yaml 配置。
"""

import logging
from datetime import datetime, timedelta

import akshare as ak
import numpy as np

logger = logging.getLogger(__name__)


def fetch_benchmark_data(code="000300", days=1000):
    """获取基准指数日线数据。

    使用 nav_cache SQLite 缓存（复用 nav_history 表，close 存为 nav 字段）。
    始终从 API 拉取最新数据并更新缓存，API 失败时降级返回缓存。

    Args:
        code: 指数代码（默认 000300 = 沪深300）
        days: 获取天数

    Returns:
        list[dict]: [{date, close}, ...] 按日期升序
    """
    import nav_cache

    try:
        df = ak.stock_zh_index_daily(symbol=f"sh{code}")
        if df is None or df.empty:
            raise ValueError("API 返回空数据")

        records = []
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        for _, row in df.iterrows():
            date_str = str(row["date"])[:10]
            if date_str >= cutoff:
                records.append({
                    "date": date_str,
                    "close": float(row["close"]),
                })

        records.sort(key=lambda x: x["date"])

        # 保存到 SQLite（close → nav 字段名转换）
        nav_records = [{"date": r["date"], "nav": r["close"]} for r in records]
        nav_cache.save_nav(code, nav_records)

        return records

    except Exception as e:
        logger.warning("获取基准数据失败: %s", e)
        # API 失败时降级返回缓存
        nav_data = nav_cache.get_nav(code, days)
        if nav_data:
            return [{"date": r["date"], "close": r["nav"]} for r in nav_data]
        return []


def compute_benchmark_stats(benchmark_data, start_date, end_date):
    """计算基准在指定区间的收益指标。

    Args:
        benchmark_data: fetch_benchmark_data() 返回的数据
        start_date: 回测起始日期 (datetime 或 str)
        end_date: 回测结束日期 (datetime 或 str)

    Returns:
        dict: {total_return, annual_return, annual_volatility, max_drawdown, sharpe}
    """
    if not benchmark_data:
        return {}

    if isinstance(start_date, datetime):
        start_str = start_date.strftime("%Y-%m-%d")
    else:
        start_str = str(start_date)[:10]

    if isinstance(end_date, datetime):
        end_str = end_date.strftime("%Y-%m-%d")
    else:
        end_str = str(end_date)[:10]

    # 筛选区间数据
    filtered = [r for r in benchmark_data if start_str <= r["date"] <= end_str]
    if len(filtered) < 2:
        return {}

    closes = np.array([r["close"] for r in filtered])
    n_days = len(closes)
    n_years = n_days / 244

    # 总收益
    total_ret = closes[-1] / closes[0] - 1

    # 年化
    annual_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # 日收益率
    daily_returns = np.diff(closes) / closes[:-1]

    # 年化波动
    annual_vol = float(np.std(daily_returns) * np.sqrt(244)) if len(daily_returns) > 1 else 0.0

    # 夏普
    sharpe = (annual_ret - 0.02) / annual_vol if annual_vol > 0 else 0.0

    # 最大回撤
    cummax = np.maximum.accumulate(closes)
    drawdowns = (cummax - closes) / cummax
    max_dd = float(np.max(drawdowns))

    return {
        "total_return": total_ret,
        "annual_return": annual_ret,
        "annual_volatility": annual_vol,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    }


def format_benchmark_comparison(strategy_stats, benchmark_stats, benchmark_name="沪深300"):
    """格式化策略 vs 基准的对比表。

    Args:
        strategy_stats: 策略的 stats 字典
        benchmark_stats: compute_benchmark_stats() 的返回值
        benchmark_name: 基准名称

    Returns:
        str: 格式化的对比文本
    """
    if not benchmark_stats:
        return ""

    lines = []
    lines.append("─── 策略 vs 基准 ({}) ───".format(benchmark_name))
    lines.append(f"  {'指标':<12s} {'策略':>10s} {'基准':>10s} {'超额':>10s}")
    lines.append(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10}")

    metrics = [
        ("总收益率", strategy_stats.get("total_return", 0), benchmark_stats.get("total_return", 0)),
        ("年化收益", strategy_stats.get("annual_return", 0), benchmark_stats.get("annual_return", 0)),
        ("年化波动", strategy_stats.get("annual_volatility", 0), benchmark_stats.get("annual_volatility", 0)),
        ("夏普比率", strategy_stats.get("sharpe", 0), benchmark_stats.get("sharpe", 0)),
        ("最大回撤", strategy_stats.get("max_drawdown", 0), benchmark_stats.get("max_drawdown", 0)),
    ]

    for name, strat_val, bench_val in metrics:
        if name in ("夏普比率",):
            excess = strat_val - bench_val
            lines.append(f"  {name:<12s} {strat_val:>10.2f} {bench_val:>10.2f} {excess:>+10.2f}")
        elif name in ("年化波动", "最大回撤"):
            excess = strat_val - bench_val
            lines.append(f"  {name:<12s} {strat_val:>10.1%} {bench_val:>10.1%} {excess:>+10.1%}")
        else:
            excess = strat_val - bench_val
            lines.append(f"  {name:<12s} {strat_val:>+10.2%} {bench_val:>+10.2%} {excess:>+10.2%}")

    return "\n".join(lines)
