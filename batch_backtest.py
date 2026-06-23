#!/usr/bin/env python3
"""200只基金批量回测：验证 TSMOM(126d)+MA100/200 策略在多窗口下的有效性。

策略逻辑 (AND 模式):
  - TSMOM(126d) > 0 AND MA100 > MA200 → 满仓
  - TSMOM(126d) < 0 AND MA100 < MA200 → 清仓
  - 信号不一致 → 维持当前仓位

性能优化：用 numpy 预计算所有指标数组，避免逐日重复计算。
"""
import sqlite3
import json
import numpy as np
import random
import os
import sys
from collections import defaultdict

DB = "/mnt/d/software/USTC-Drive/科大云盘/charming-cloud/project/quant-fund/cache/nav_cache.db"
FEE_BUY, FEE_SELL = 0.0015, 0.005
INITIAL_CAPITAL = 100000.0

# 时间窗口
WINDOWS = [
    ("3yr_2023_2026", "2023-06-18", "2026-06-18"),
    ("5yr_2021_2026", "2021-07-01", "2026-06-18"),
    ("3yr_2021_2024", "2021-07-01", "2024-06-18"),
]


def load_funds():
    """加载选中的200只基金列表和净值数据"""
    with open("/mnt/d/software/USTC-Drive/科大云盘/charming-cloud/project/quant-fund/sample200.json", "r", encoding="utf-8") as f:
        funds = json.load(f)

    conn = sqlite3.connect(DB)
    fund_data = {}
    for fund in funds:
        code = fund["code"]
        rows = conn.execute(
            "SELECT date, nav FROM nav_history WHERE code=? ORDER BY date", (code,)
        ).fetchall()
        if len(rows) > 200:
            fund_data[code] = {
                "code": code,
                "name": fund.get("name", code),
                "type": fund.get("type", ""),
                "records": [{"date": r[0], "nav": r[1]} for r in rows],
            }
    conn.close()
    return list(fund_data.values())


def precompute_indicators(records):
    """预计算 MA100, MA200, TSMOM126 数组（向量化）。"""
    navs = np.array([r["nav"] for r in records])
    dates = [r["date"] for r in records]
    n = len(navs)

    # MA200 (cumsum 法)
    ma200 = np.full(n, np.nan)
    ma100 = np.full(n, np.nan)
    cumsum = np.cumsum(navs)
    for i in range(199, n):
        if i >= 199:
            ma200[i] = (cumsum[i] - (cumsum[i-200] if i >= 200 else 0)) / 200
        if i >= 99:
            ma100[i] = (cumsum[i] - (cumsum[i-100] if i >= 100 else 0)) / 100

    # TSMOM 126d
    tsmom126 = np.zeros(n)
    for i in range(126, n):
        tsmom126[i] = navs[i] / navs[i-126] - 1

    return dates, navs, ma100, ma200, tsmom126


def backtest_fund(records, start_date, end_date, strategy="AND"):
    """单基金回测（使用预计算指标）。

    strategy:
      "AND"    — TSMOM>0 AND MA100>MA200 → full; both bear → cash; mixed → hold
      "AND_half" — same but mixed → 50%
      "OR"     — any bull → full; both bear → cash
      "tsmom_only" — TSMOM>0 → full; <0 → cash
    """
    dates, navs, ma100, ma200, tsmom126 = precompute_indicators(records)

    # 找回测区间
    start_idx = None
    end_idx = len(dates) - 1
    for i, d in enumerate(dates):
        if d >= start_date and start_idx is None:
            start_idx = i
            break
    if start_idx is None or start_idx < 200:
        return None

    for i in range(start_idx, len(dates)):
        if dates[i] > end_date:
            end_idx = i - 1
            break

    if end_idx <= start_idx:
        return None

    # Buy & Hold
    nav0 = navs[start_idx]
    nav_end = navs[end_idx]
    if nav0 <= 0:
        return None
    bh_return = nav_end / nav0 - 1

    # 策略回测
    cash = INITIAL_CAPITAL
    units = 0.0
    pos = 0.0
    trades = 0

    for i in range(start_idx, end_idx + 1):
        nav = navs[i]
        if nav <= 0 or np.isnan(ma200[i]) or np.isnan(ma100[i]):
            continue

        mom = tsmom126[i]
        golden = ma100[i] > ma200[i]
        tsmom_bull = mom > 0

        if strategy == "AND":
            if tsmom_bull and golden:
                target = 1.0
            elif not tsmom_bull and not golden:
                target = 0.0
            else:
                target = pos  # hold
        elif strategy == "AND_half":
            if tsmom_bull and golden:
                target = 1.0
            elif not tsmom_bull and not golden:
                target = 0.0
            else:
                target = 0.5
        elif strategy == "OR":
            if tsmom_bull or golden:
                target = 1.0
            else:
                target = 0.0
        elif strategy == "tsmom_only":
            target = 1.0 if tsmom_bull else 0.0
        else:
            target = pos

        target = max(0.0, min(1.0, target))

        tv = cash + units * nav
        diff = tv * target - units * nav
        if abs(diff) < tv * 0.05:
            continue

        if diff > 0 and cash > 100:
            ba = min(diff, cash * 0.98)
            units += (ba - ba * FEE_BUY) / nav
            cash -= ba
            trades += 1
        elif diff < 0 and units > 0:
            su = min(units * abs(diff) / (units * nav), units)
            cash += su * nav * (1 - FEE_SELL)
            units -= su
            trades += 1

        pos = units * nav / (cash + units * nav) if (cash + units * nav) > 0 else 0

    final_value = cash + units * navs[end_idx]
    strat_return = final_value / INITIAL_CAPITAL - 1

    return {
        "strat_return": strat_return,
        "bh_return": bh_return,
        "excess": strat_return - bh_return,
        "trades": trades,
    }


def main():
    funds = load_funds()
    print(f"加载 {len(funds)} 只基金")

    strategies = ["AND", "AND_half", "OR", "tsmom_only"]

    all_results = {}

    for wname, start, end in WINDOWS:
        print(f"\n{'='*70}")
        print(f"窗口: {wname} ({start} ~ {end})")
        print(f"{'='*70}")

        for sname in strategies:
            results = []
            for fund in funds:
                r = backtest_fund(fund["records"], start, end, strategy=sname)
                if r is not None:
                    r["code"] = fund["code"]
                    r["name"] = fund["name"]
                    r["type"] = fund["type"]
                    results.append(r)

            if not results:
                continue

            wins = [r for r in results if r["excess"] > 0]
            losses = [r for r in results if r["excess"] <= 0]
            avg_strat = np.mean([r["strat_return"] for r in results])
            avg_bh = np.mean([r["bh_return"] for r in results])
            avg_excess = np.mean([r["excess"] for r in results])
            avg_trades = np.mean([r["trades"] for r in results])
            median_excess = np.median([r["excess"] for r in results])

            print(f"\n  策略: {sname}")
            print(f"    有效基金: {len(results)}")
            print(f"    胜率: {len(wins)}/{len(results)} = {len(wins)/len(results)*100:.1f}%")
            print(f"    平均策略收益: {avg_strat:+.2%}")
            print(f"    平均持有收益: {avg_bh:+.2%}")
            print(f"    平均超额收益: {avg_excess:+.2%}")
            print(f"    中位超额收益: {median_excess:+.2%}")
            print(f"    平均交易次数: {avg_trades:.1f}")

            key = f"{wname}_{sname}"
            all_results[key] = {
                "window": wname,
                "strategy": sname,
                "n": len(results),
                "wins": len(wins),
                "win_rate": len(wins) / len(results),
                "avg_strat": avg_strat,
                "avg_bh": avg_bh,
                "avg_excess": avg_excess,
                "median_excess": median_excess,
                "avg_trades": avg_trades,
                "results": results,
            }

    # 保存完整结果
    with open("/tmp/batch_results.json", "w", encoding="utf-8") as f:
        # 不保存 results 内的详细数据，只保存汇总
        summary = {k: {kk: vv for kk, vv in v.items() if kk != "results"} for k, v in all_results.items()}
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到 /tmp/batch_results.json")

    # 保存详细结果（用于分析）
    with open("/tmp/batch_detail.json", "w", encoding="utf-8") as f:
        detail = {}
        for k, v in all_results.items():
            detail[k] = v["results"]
        json.dump(detail, f, ensure_ascii=False, indent=2, default=str)
    print(f"详细结果已保存到 /tmp/batch_detail.json")

    # 汇总表
    print(f"\n\n{'='*90}")
    print(f"  汇总表")
    print(f"{'='*90}")
    print(f"  {'窗口':<18} {'策略':<14} {'N':>4} {'胜率':>7} {'平均策略':>9} {'平均持有':>9} {'平均超额':>9} {'中位超额':>9} {'交易':>5}")
    print(f"  {'-'*18} {'-'*14} {'-'*4} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*5}")
    for wname, _, _ in WINDOWS:
        for sname in strategies:
            key = f"{wname}_{sname}"
            if key not in all_results:
                continue
            v = all_results[key]
            print(
                f"  {v['window']:<18} {v['strategy']:<14} {v['n']:>4} "
                f"{v['win_rate']*100:>6.1f}% "
                f"{v['avg_strat']:>+8.2%} {v['avg_bh']:>+8.2%} "
                f"{v['avg_excess']:>+8.2%} {v['median_excess']:>+8.2%} "
                f"{v['avg_trades']:>5.1f}"
            )
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
