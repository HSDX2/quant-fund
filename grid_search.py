#!/usr/bin/env python3
"""P1-P8/S1-S8 规则参数网格搜索。

在核心+卫星+MA200过滤框架下，用 P1-P8 做卫星买入信号，
对规则参数做网格搜索，找到使多数基金（含沪深300）在多窗口跑赢持有的最优参数组。

搜索策略：分组搜索（先 RSI，再连跌，再卖出参数），避免组合爆炸。
"""
import json
import numpy as np
from collections import defaultdict
from swing_backtest import (
    precompute, check_buy, check_sell, load_funds,
    BENCHMARK, WINDOWS, FEE_BUY, FEE_SELL, INITIAL_CAPITAL,
    DEFAULT_BUY_RULES, DEFAULT_SELL_RULES,
)


def backtest_core_satellite(ind, start_date, end_date, buy_rules, sell_rules, params):
    """核心+卫星策略：核心持有，卫星用P1-P8/S1-S8做短线波段。

    核心仓位（core_ratio）：买入持有永不卖出
    卫星仓位（1-core_ratio）：
      - MA200上方：保持满仓（牛市不交易）
      - MA200下方：P1-P8买入 + 短线退出（中轨止盈/超时/S1清仓）
    """
    if ind is None:
        return None

    dates = ind["dates"]
    navs = ind["navs"]
    n = len(navs)

    core_ratio = params.get("core_ratio", 0.3)
    max_hold = params.get("max_hold_days", 14)
    min_hold = params.get("min_hold_days", 7)
    cooldown = params.get("buy_cooldown", 2)
    trailing_stop = params.get("trailing_stop", 0.08)  # 移动止盈
    stop_loss = params.get("stop_loss", -0.08)          # 止损

    # 定位回测区间
    start_idx = None
    for i, d in enumerate(dates):
        if d >= start_date:
            start_idx = i
            break
    if start_idx is None or start_idx < 200:
        return None
    end_idx = n - 1
    for i in range(start_idx, n):
        if dates[i] > end_date:
            end_idx = i - 1
            break
    if end_idx <= start_idx:
        return None

    nav0 = navs[start_idx]
    nav_end = navs[end_idx]
    if nav0 <= 0:
        return None
    bh_return = (1 - FEE_BUY) * nav_end / nav0 - 1

    ma200 = ind["ma200"]
    bb_mid = ind["bb_mid"]  # MA20，用于中轨止盈

    cash = INITIAL_CAPITAL
    core_units = 0.0
    swing_units = 0.0
    trades = 0
    last_sell = -999
    last_buy = -999
    buy_nav = 0.0
    peak_nav = 0.0
    trade_log = []

    # 起始：核心+卫星都满仓
    if nav0 > 0:
        core_amt = cash * core_ratio * 0.999
        core_fee = core_amt * FEE_BUY
        core_units = (core_amt - core_fee) / nav0
        cash -= core_amt

        swing_amt = cash * 0.999
        swing_fee = swing_amt * FEE_BUY
        swing_units = (swing_amt - swing_fee) / nav0
        cash -= swing_amt
        last_buy = start_idx
        buy_nav = nav0
        peak_nav = nav0

    for i in range(start_idx, end_idx + 1):
        nav = navs[i]
        if nav <= 0 or np.isnan(ma200[i]):
            continue

        above_ma = nav >= ma200[i]

        if swing_units > 0.01:
            # === 卫星持仓 ===
            if nav > peak_nav:
                peak_nav = nav

            days_held = i - last_buy
            pnl = (nav - buy_nav) / buy_nav if buy_nav > 0 else 0
            drawdown = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0
            exit_reason = None

            # MA200上方（牛市）：卫星不卖
            if not above_ma:
                # MA200下方（非牛市）：短线退出规则
                # 重置持有天数（刚从牛市切换过来时）
                if days_held > max_hold * 3:
                    last_buy = i
                    days_held = 0
                    peak_nav = nav

                # 1. 止损（不受 min_hold 限制）
                if pnl <= stop_loss:
                    exit_reason = f"止损({pnl:+.1%})"
                elif days_held >= min_hold:
                    # 2. 移动止盈
                    if drawdown >= trailing_stop:
                        exit_reason = f"移动止盈({pnl:+.1%},DD{drawdown:.1%})"
                    # 3. 回到中轨止盈
                    elif not np.isnan(bb_mid[i]) and nav >= bb_mid[i]:
                        exit_reason = f"回到中轨({pnl:+.1%})"
                    # 4. 超时退出
                    elif days_held >= max_hold:
                        exit_reason = f"超时({days_held}d,{pnl:+.1%})"
                    # 5. S1 硬止损
                    else:
                        action, srule = check_sell(i, ind, sell_rules)
                        if action == "清仓":
                            exit_reason = srule

            if exit_reason and days_held >= 1:
                sell_fee = FEE_SELL if days_held >= 7 else 0.015
                gross = swing_units * nav
                fee = gross * sell_fee
                cash += gross - fee
                swing_units = 0.0
                trades += 1
                last_sell = i
                trade_log.append((dates[i], "SELL", nav, exit_reason))

        else:
            # === 卫星空仓 ===
            if above_ma:
                # 牛市回归：立即满仓
                if cash > 100:
                    buy_amt = cash * 0.999
                    fee = buy_amt * FEE_BUY
                    swing_units = (buy_amt - fee) / nav
                    cash -= buy_amt
                    trades += 1
                    last_buy = i
                    buy_nav = nav
                    peak_nav = nav
                    trade_log.append((dates[i], "BUY", nav, "牛市回归"))
            else:
                # 非牛市：P1-P8 规则买入
                if i - last_sell < cooldown:
                    continue
                rule = check_buy(i, ind, buy_rules)
                if rule and cash > 100:
                    buy_amt = cash * 0.999
                    fee = buy_amt * FEE_BUY
                    swing_units = (buy_amt - fee) / nav
                    cash -= buy_amt
                    trades += 1
                    last_buy = i
                    buy_nav = nav
                    peak_nav = nav
                    trade_log.append((dates[i], "BUY", nav, rule))

    final_value = cash + (core_units + swing_units) * navs[end_idx]
    strat_return = final_value / INITIAL_CAPITAL - 1

    return {
        "strat_return": strat_return,
        "bh_return": bh_return,
        "excess": strat_return - bh_return,
        "trades": trades,
        "trade_log": trade_log,
    }


def evaluate(funds, precomputed, buy_rules, sell_rules, params):
    """评估一组参数在所有基金×所有窗口上的表现。"""
    wnames = [w[0] for w in WINDOWS]
    results = {w[0]: {} for w in WINDOWS}

    for code, ind in precomputed.items():
        for wname, sdate, edate in WINDOWS:
            r = backtest_core_satellite(ind, sdate, edate, buy_rules, sell_rules, params)
            if r:
                results[wname][code] = r

    # 统计
    fund_wins = defaultdict(list)
    for wn in wnames:
        for code, r in results[wn].items():
            fund_wins[code].append(r["excess"] > 0)
    full = {c: w for c, w in fund_wins.items() if len(w) == len(wnames)}
    if not full:
        return None

    win2 = sum(1 for w in full.values() if sum(w) >= 2)
    non_bull_win = sum(1 for w in full.values() if w[2] and w[3])  # 近5年+熊市3年
    bull_exc = np.mean([r["excess"] for wn in [wnames[0], wnames[1]] for r in results[wn].values()])
    non_bull_exc = np.mean([r["excess"] for wn in [wnames[2], wnames[3]] for r in results[wn].values()])
    bm_exc = [results[wn].get(BENCHMARK, {}).get("excess", 0) for wn in wnames]
    bm_non_bull = bm_exc[2] + bm_exc[3]  # CSI300 非牛市超额之和

    return {
        "win2": win2,
        "total": len(full),
        "win2_pct": win2 / len(full),
        "non_bull_win": non_bull_win,
        "non_bull_pct": non_bull_win / len(full),
        "bull_exc": bull_exc,
        "non_bull_exc": non_bull_exc,
        "bm_exc": bm_exc,
        "bm_non_bull": bm_non_bull,
    }


def main():
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(f"加载 {n} 只基金...", flush=True)
    funds = load_funds(n=n, include_benchmark=True)

    # 预计算指标
    precomputed = {}
    for fund in funds:
        ind = precompute(fund["records"])
        if ind is not None:
            precomputed[fund["code"]] = ind
    print(f"预计算 {len(precomputed)} 只基金", flush=True)

    # 基础退出参数（固定）
    base_params = {
        "core_ratio": 0.3,
        "max_hold_days": 14,
        "min_hold_days": 7,
        "buy_cooldown": 2,
        "trailing_stop": 0.08,
        "stop_loss": -0.08,
    }

    # ── 第1轮：搜 RSI 类参数 ──
    # P1/P2/P4/P5 都有 rsi_max，统一搜索
    print("\n=== 第1轮：RSI 参数搜索 ===", flush=True)
    rsi_grid = [30, 35, 40, 45, 50]
    best_rsi = None
    best_score_rsi = -999

    for rsi_p1 in rsi_grid:
        for rsi_p2 in [30, 35, 40]:
            for rsi_p4 in [35, 40, 45]:
                buy_rules = {
                    "P1": {"rsi_max": rsi_p1, "decline_min": 2, "decline_max": 7},
                    "P2": {"rsi_max": rsi_p2, "max_dd_min": 0.06},
                    "P3": {"decline_min": 2, "decline_max": 7, "sharpe_min": 0.3},
                    "P4": {"rsi_max": rsi_p4, "vol_ratio_max": 1.0},
                    "P5": {"max_dd_min": 0.10, "rsi_max": rsi_p4},
                    "P6": {"sharpe_min": 0.5, "decline_min": 1, "decline_max": 7},
                    "P7": {"vol_ratio_max": 0.6, "pullback_min": 0.02},
                    "P8": {"trend_min": 0.03, "decline_min": 1},
                }
                sell_rules = {**DEFAULT_SELL_RULES, "S1": {"max_dd_min": 0.15, "action": "清仓"}}
                ev = evaluate(funds, precomputed, buy_rules, sell_rules, base_params)
                if ev is None:
                    continue
                # 评分：非牛市超额为主，牛市不亏太多
                score = ev["non_bull_exc"] * 2 + ev["bull_exc"] + ev["bm_non_bull"] * 0.5
                if score > best_score_rsi:
                    best_score_rsi = score
                    best_rsi = (rsi_p1, rsi_p2, rsi_p4)
                    print(f"  RSI P1={rsi_p1} P2={rsi_p2} P4={rsi_p4}: "
                          f"bull={ev['bull_exc']:+.1%} nonbull={ev['non_bull_exc']:+.1%} "
                          f"CSI300_nb={ev['bm_non_bull']:+.1%} win2={ev['win2_pct']:.0%} "
                          f"score={score:+.2f} ← NEW BEST", flush=True)

    print(f"\n第1轮最优 RSI: P1={best_rsi[0]} P2={best_rsi[1]} P4={best_rsi[2]}", flush=True)

    # ── 第2轮：搜连跌/夏普/回撤参数 ──
    print("\n=== 第2轮：连跌/夏普/回撤参数搜索 ===", flush=True)
    rsi_p1, rsi_p2, rsi_p4 = best_rsi
    best_params2 = None
    best_score2 = -999

    for dec_min in [1, 2, 3]:
        for dec_max in [5, 7, 10]:
            for sharpe_p3 in [0.0, 0.3, 0.5]:
                for dd_p2 in [0.04, 0.06, 0.08]:
                    buy_rules = {
                        "P1": {"rsi_max": rsi_p1, "decline_min": dec_min, "decline_max": dec_max},
                        "P2": {"rsi_max": rsi_p2, "max_dd_min": dd_p2},
                        "P3": {"decline_min": dec_min, "decline_max": dec_max, "sharpe_min": sharpe_p3},
                        "P4": {"rsi_max": rsi_p4, "vol_ratio_max": 1.0},
                        "P5": {"max_dd_min": 0.10, "rsi_max": rsi_p4},
                        "P6": {"sharpe_min": 0.5, "decline_min": dec_min, "decline_max": dec_max},
                        "P7": {"vol_ratio_max": 0.6, "pullback_min": 0.02},
                        "P8": {"trend_min": 0.03, "decline_min": dec_min},
                    }
                    sell_rules = {**DEFAULT_SELL_RULES, "S1": {"max_dd_min": 0.15, "action": "清仓"}}
                    ev = evaluate(funds, precomputed, buy_rules, sell_rules, base_params)
                    if ev is None:
                        continue
                    score = ev["non_bull_exc"] * 2 + ev["bull_exc"] + ev["bm_non_bull"] * 0.5
                    if score > best_score2:
                        best_score2 = score
                        best_params2 = (dec_min, dec_max, sharpe_p3, dd_p2)
                        print(f"  dec={dec_min}-{dec_max} sharpe={sharpe_p3} dd={dd_p2}: "
                              f"bull={ev['bull_exc']:+.1%} nonbull={ev['non_bull_exc']:+.1%} "
                              f"CSI300_nb={ev['bm_non_bull']:+.1%} win2={ev['win2_pct']:.0%} "
                              f"score={score:+.2f} ← NEW BEST", flush=True)

    print(f"\n第2轮最优: dec={best_params2[0]}-{best_params2[1]} sharpe={best_params2[2]} dd={best_params2[3]}", flush=True)

    # ── 第3轮：搜卖出参数（S1回撤阈值 + 退出参数）──
    print("\n=== 第3轮：卖出参数搜索 ===", flush=True)
    dec_min, dec_max, sharpe_p3, dd_p2 = best_params2
    best_params3 = None
    best_score3 = -999

    for s1_dd in [0.10, 0.12, 0.15, 0.20]:
        for trailing in [0.05, 0.08, 0.10, 0.12]:
            for stop_l in [-0.05, -0.08, -0.10]:
                for max_hold in [10, 14, 21]:
                    buy_rules = {
                        "P1": {"rsi_max": rsi_p1, "decline_min": dec_min, "decline_max": dec_max},
                        "P2": {"rsi_max": rsi_p2, "max_dd_min": dd_p2},
                        "P3": {"decline_min": dec_min, "decline_max": dec_max, "sharpe_min": sharpe_p3},
                        "P4": {"rsi_max": rsi_p4, "vol_ratio_max": 1.0},
                        "P5": {"max_dd_min": 0.10, "rsi_max": rsi_p4},
                        "P6": {"sharpe_min": 0.5, "decline_min": dec_min, "decline_max": dec_max},
                        "P7": {"vol_ratio_max": 0.6, "pullback_min": 0.02},
                        "P8": {"trend_min": 0.03, "decline_min": dec_min},
                    }
                    sell_rules = {**DEFAULT_SELL_RULES, "S1": {"max_dd_min": s1_dd, "action": "清仓"}}
                    params = {**base_params, "trailing_stop": trailing, "stop_loss": stop_l, "max_hold_days": max_hold}
                    ev = evaluate(funds, precomputed, buy_rules, sell_rules, params)
                    if ev is None:
                        continue
                    score = ev["non_bull_exc"] * 2 + ev["bull_exc"] + ev["bm_non_bull"] * 0.5
                    if score > best_score3:
                        best_score3 = score
                        best_params3 = (s1_dd, trailing, stop_l, max_hold)
                        print(f"  S1_dd={s1_dd} trail={trailing} stop={stop_l} mh={max_hold}: "
                              f"bull={ev['bull_exc']:+.1%} nonbull={ev['non_bull_exc']:+.1%} "
                              f"CSI300_nb={ev['bm_non_bull']:+.1%} win2={ev['win2_pct']:.0%} "
                              f"score={score:+.2f} ← NEW BEST", flush=True)

    print(f"\n第3轮最优: S1_dd={best_params3[0]} trail={best_params3[1]} stop={best_params3[2]} mh={best_params3[3]}", flush=True)

    # ── 最终结果 ──
    s1_dd, trailing, stop_l, max_hold = best_params3
    final_buy = {
        "P1": {"rsi_max": rsi_p1, "decline_min": dec_min, "decline_max": dec_max},
        "P2": {"rsi_max": rsi_p2, "max_dd_min": dd_p2},
        "P3": {"decline_min": dec_min, "decline_max": dec_max, "sharpe_min": sharpe_p3},
        "P4": {"rsi_max": rsi_p4, "vol_ratio_max": 1.0},
        "P5": {"max_dd_min": 0.10, "rsi_max": rsi_p4},
        "P6": {"sharpe_min": 0.5, "decline_min": dec_min, "decline_max": dec_max},
        "P7": {"vol_ratio_max": 0.6, "pullback_min": 0.02},
        "P8": {"trend_min": 0.03, "decline_min": dec_min},
    }
    final_sell = {**DEFAULT_SELL_RULES, "S1": {"max_dd_min": s1_dd, "action": "清仓"}}
    final_params = {**base_params, "trailing_stop": trailing, "stop_loss": stop_l, "max_hold_days": max_hold}

    ev = evaluate(funds, precomputed, final_buy, final_sell, final_params)
    print(f"\n{'='*60}")
    print(f"最终最优参数组合")
    print(f"{'='*60}")
    print(f"买入规则:")
    for k, v in final_buy.items():
        print(f"  {k}: {v}")
    print(f"卖出规则:")
    for k, v in final_sell.items():
        print(f"  {k}: {v}")
    print(f"退出参数: trailing={trailing} stop={stop_l} max_hold={max_hold} core={base_params['core_ratio']}")
    print(f"\n结果:")
    print(f"  ≥2窗口赢: {ev['win2']}/{ev['total']} = {ev['win2_pct']:.1%}")
    print(f"  非牛市双赢: {ev['non_bull_win']}/{ev['total']} = {ev['non_bull_pct']:.1%}")
    print(f"  牛市平均超额: {ev['bull_exc']:+.2%}")
    print(f"  非牛市平均超额: {ev['non_bull_exc']:+.2%}")
    print(f"  沪深300各窗口: {[f'{e:+.1%}' for e in ev['bm_exc']]}")

    # 保存结果
    out = {
        "buy_rules": final_buy,
        "sell_rules": final_sell,
        "params": final_params,
        "results": {
            "win2_pct": ev["win2_pct"],
            "non_bull_pct": ev["non_bull_pct"],
            "bull_exc": ev["bull_exc"],
            "non_bull_exc": ev["non_bull_exc"],
            "bm_exc": ev["bm_exc"],
        },
    }
    with open("/mnt/d/software/USTC-Drive/科大云盘/charming-cloud/project/quant-fund/grid_search_result.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: grid_search_result.json")


if __name__ == "__main__":
    main()
