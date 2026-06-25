#!/usr/bin/env python3
"""Swing trading 参数扫描：在含沪深300的小样本上找最优退出参数。"""
import sys
sys.path.insert(0, "/mnt/d/software/USTC-Drive/科大云盘/charming-cloud/project/quant-fund")
from swing_backtest import (
    load_funds, backtest_fund, backtest_fund_with_ind, DEFAULT_BUY_RULES, DEFAULT_SELL_RULES,
    BENCHMARK, FEE_BUY, FEE_SELL
)
import numpy as np
from itertools import product

WINDOWS = [
    ("近1年", "2025-06-18", "2026-06-18"),
    ("近3年", "2023-06-18", "2026-06-18"),
    ("近5年", "2021-06-18", "2026-06-18"),
]

def make_exit(trailing, stop_loss, max_hold, cooldown, min_hold=3):
    return {
        "trailing_stop": trailing,
        "stop_loss": stop_loss,
        "max_hold_days": max_hold,
        "buy_cooldown": cooldown,
        "min_hold_days": min_hold,
    }

def eval_params(funds, exit_params, buy_rules, sell_rules, precomputed=None):
    """评估一组参数在所有基金×所有窗口的表现。返回汇总指标。"""
    results = {w[0]: {} for w in WINDOWS}
    for fund in funds:
        code = fund["code"]
        # 预计算指标（可复用）
        if precomputed is not None and code in precomputed:
            ind = precomputed[code]
        else:
            from swing_backtest import precompute
            ind = precompute(fund["records"])
            if precomputed is not None:
                precomputed[code] = ind
        if ind is None:
            continue
        for wname, sdate, edate in WINDOWS:
            r = backtest_fund_with_ind(ind, sdate, edate, buy_rules, sell_rules, exit_params)
            if r:
                results[wname][code] = r

    # 汇总
    total_wins = 0
    total_n = 0
    bm_wins = 0
    bm_total = 0
    avg_excess_all = []

    for wname, cr in results.items():
        if not cr:
            continue
        wins = sum(1 for r in cr.values() if r["excess"] > 0)
        total_wins += wins
        total_n += len(cr)
        avg_excess_all.extend([r["excess"] for r in cr.values()])
        bm = cr.get(BENCHMARK)
        if bm:
            bm_total += 1
            if bm["excess"] > 0:
                bm_wins += 1

    overall_winrate = total_wins / total_n if total_n > 0 else 0
    bm_winrate = bm_wins / bm_total if bm_total > 0 else 0
    avg_excess = np.mean(avg_excess_all) if avg_excess_all else 0

    return {
        "winrate": overall_winrate,
        "bm_winrate": bm_winrate,
        "avg_excess": avg_excess,
        "total_wins": total_wins,
        "total_n": total_n,
        "results": results,
    }


def main():
    print("加载基金数据(20只+沪深300)...", flush=True)
    funds = load_funds(n=20, include_benchmark=True)
    print(f"共 {len(funds)} 只基金", flush=True)

    # 预计算所有基金指标（只算一次，复用）
    from swing_backtest import precompute
    print("预计算指标...", flush=True)
    precomputed = {}
    for fund in funds:
        ind = precompute(fund["records"])
        if ind is not None:
            precomputed[fund["code"]] = ind
    print(f"已预计算 {len(precomputed)} 只基金指标", flush=True)

    # 参数网格
    trailings = [0.05, 0.08, 0.10, 0.12, 0.15]
    stop_losses = [-0.05, -0.08, -0.10]
    max_holds = [10, 15, 20, 30, 999]
    cooldowns = [3, 5]

    best = None
    best_score = -999
    results_list = []

    total_combos = len(trailings) * len(stop_losses) * len(max_holds) * len(cooldowns)
    print(f"\n扫描 {total_combos} 种参数组合...", flush=True)

    for i, (tr, sl, mh, cd) in enumerate(product(trailings, stop_losses, max_holds, cooldowns)):
        exit_params = make_exit(tr, sl, mh, cd)
        r = eval_params(funds, exit_params, DEFAULT_BUY_RULES, DEFAULT_SELL_RULES, precomputed)

        # 评分：综合胜率 + 平均超额 + 基准胜率
        score = r["winrate"] * 100 + r["avg_excess"] * 100 + r["bm_winrate"] * 50
        results_list.append({
            "trailing": tr, "stop_loss": sl, "max_hold": mh, "cooldown": cd,
            "winrate": r["winrate"], "bm_winrate": r["bm_winrate"],
            "avg_excess": r["avg_excess"],
            "score": score,
        })

        if r["winrate"] > 0.5 and r["bm_winrate"] >= 0.5:
            if score > best_score:
                best_score = score
                best = (tr, sl, mh, cd, r)

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{total_combos}", flush=True)

    # 排序输出 TOP 15
    results_list.sort(key=lambda x: x["score"], reverse=True)
    print("\n" + "=" * 100)
    print(f"{'trailing':>8} {'stop_loss':>10} {'max_hold':>9} {'cooldown':>8} | {'winrate':>8} {'bm_wr':>6} {'avg_excess':>11} {'score':>7}")
    print("-" * 100)
    for r in results_list[:15]:
        mh_str = "∞" if r["max_hold"] == 999 else str(r["max_hold"])
        print(f"{r['trailing']:>8.2f} {r['stop_loss']:>10.2f} {mh_str:>9} {r['cooldown']:>8} | "
              f"{r['winrate']:>7.1%} {r['bm_winrate']:>6.1%} {r['avg_excess']:>+10.2%} {r['score']:>7.1f}")

    # 输出满足条件的组合
    print("\n" + "=" * 100)
    qualified = [r for r in results_list if r["winrate"] > 0.5 and r["bm_winrate"] >= 0.5]
    if qualified:
        print(f"满足条件(胜率>50%且沪深300赢)的组合: {len(qualified)} 个")
        for r in qualified[:5]:
            mh_str = "∞" if r["max_hold"] == 999 else str(r["max_hold"])
            print(f"  trailing={r['trailing']:.2f} stop={r['stop_loss']:.2f} "
                  f"max_hold={mh_str} cooldown={r['cooldown']} → "
                  f"胜率={r['winrate']:.1%} 沪深300={r['bm_winrate']:.1%} 超额={r['avg_excess']:+.2%}")
    else:
        print("无组合满足条件(胜率>50%且沪深300赢)")
        print("最优组合:")
        r = results_list[0]
        mh_str = "∞" if r["max_hold"] == 999 else str(r["max_hold"])
        print(f"  trailing={r['trailing']:.2f} stop={r['stop_loss']:.2f} "
              f"max_hold={mh_str} cooldown={r['cooldown']} → "
              f"胜率={r['winrate']:.1%} 沪深300={r['bm_winrate']:.1%} 超额={r['avg_excess']:+.2%}")


if __name__ == "__main__":
    main()
