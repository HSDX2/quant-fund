#!/usr/bin/env python3
"""Swing trading 回测：MA200 非对称确认择时策略验证。

策略逻辑（MA200 非对称确认择时）:
  - 起始满仓（捕获牛市涨幅）
  - 退出：跌破 MA200 连续 12 天确认 → 清仓（慢退出，过滤假信号）
  - 重新入场：站上 MA200 连续 1 天确认 → 满仓买回（快入场，抓住反弹）
  - 非对称确认是核心：慢退出减少震荡假信号，快入场避免踏空

性能优化：numpy 向量化预计算所有指标数组，避免逐日重复计算。
"""
import sqlite3
import json
import numpy as np
import os
import sys
from collections import defaultdict

DB = "/mnt/d/software/USTC-Drive/科大云盘/charming-cloud/project/quant-fund/cache/nav_cache.db"
SAMPLE = "/mnt/d/software/USTC-Drive/科大云盘/charming-cloud/project/quant-fund/sample200.json"
FEE_BUY, FEE_SELL = 0.0015, 0.005
INITIAL_CAPITAL = 100000.0
BENCHMARK = "110020"  # 易方达沪深300ETF联接A

# 时间窗口（覆盖牛市、熊市、完整周期）
WINDOWS = [
    ("近3年", "2023-06-18", "2026-06-18"),
    ("近4年", "2022-06-18", "2026-06-18"),
    ("近5年", "2021-06-18", "2026-06-18"),
    ("熊市3年", "2021-06-18", "2024-06-18"),
]


# ── 规则参数（可被外部覆盖）──────────────────────────────
DEFAULT_BUY_RULES = {
    "P1": {"rsi_max": 45, "decline_min": 2, "decline_max": 7},
    "P2": {"rsi_max": 35, "max_dd_min": 0.06},
    "P3": {"decline_min": 2, "decline_max": 7, "sharpe_min": 0.3},
    "P4": {"rsi_max": 40, "vol_ratio_max": 1.0},
    "P5": {"max_dd_min": 0.10, "rsi_max": 40},
    "P6": {"sharpe_min": 0.5, "decline_min": 1, "decline_max": 7},
    "P7": {"vol_ratio_max": 0.6, "pullback_min": 0.02},
    "P8": {"trend_min": 0.03, "decline_min": 1},
}

DEFAULT_SELL_RULES = {
    "S1": {"max_dd_min": 0.15, "action": "清仓"},
    "S2": {"rise_min": 5, "rsi_min": 75, "action": "减仓", "enabled": False},
    "S3": {"monthly_min": 0.12, "rsi_min": 80, "action": "减仓", "enabled": False},
    "S4": {"sharpe_max": -1.0, "monthly_max": -0.10, "action": "减仓", "enabled": False},
    "S5": {"pullback_min": 0.08, "decline_min": 5, "action": "减仓", "enabled": False},
    "S6": {"trend_min": 0.15, "rsi_min": 70, "action": "减仓", "enabled": False},
    "S7": {"vol_ratio_min": 2.0, "action": "减仓", "enabled": False},
    "S8": {"monthly_max": -0.08, "action": "清仓", "enabled": False},
}

# MA200 非对称确认择时参数
SWING_EXIT = {
    "ma_period": 200,          # 使用 MA200 作为趋势线
    "exit_confirm_days": 12,   # 跌破 MA200 连续 12 天确认退出（慢退出）
    "entry_confirm_days": 1,   # 站上 MA200 连续 1 天确认重新入场（快入场）
    "buffer": 0.0,             # 无缓冲带
    "buy_cooldown": 3,         # 卖出后冷却 3 天
    "min_hold_days": 5,        # 最少持有 5 天
}


# ── 向量化指标计算 ──────────────────────────────────────

def precompute(records):
    """预计算所有技术指标数组。返回 dict of arrays，长度 = len(records)。"""
    navs = np.array([r["nav"] for r in records], dtype=float)
    dates = [r["date"] for r in records]
    n = len(navs)
    if n < 200:
        return None

    # 移动平均（cumsum 法）
    cumsum = np.cumsum(navs)
    ma200 = np.full(n, np.nan)
    ma100 = np.full(n, np.nan)
    ma50 = np.full(n, np.nan)
    for i in range(n):
        if i >= 199:
            prev = cumsum[i - 200] if i >= 200 else 0
            ma200[i] = (cumsum[i] - prev) / 200
        if i >= 99:
            prev = cumsum[i - 100] if i >= 100 else 0
            ma100[i] = (cumsum[i] - prev) / 100
        if i >= 49:
            prev = cumsum[i - 50] if i >= 50 else 0
            ma50[i] = (cumsum[i] - prev) / 50

    # RSI(14)
    rsi = np.full(n, 50.0)
    for i in range(14, n):
        window = navs[i - 14:i + 1]
        deltas = np.diff(window)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    # 滚动夏普 (60d)
    sharpe = np.zeros(n)
    for i in range(60, n):
        window = navs[i - 60:i + 1]
        returns = np.diff(window) / window[:-1]
        if len(returns) < 2 or np.std(returns) == 0:
            sharpe[i] = 0.0
        else:
            ann_ret = (window[-1] / window[0]) ** (244 / 60) - 1
            ann_vol = np.std(returns, ddof=1) * np.sqrt(244)
            sharpe[i] = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0.0

    # 滚动最大回撤 (60d)
    max_dd = np.zeros(n)
    for i in range(60, n):
        window = navs[i - 60:i + 1]
        cummax = np.maximum.accumulate(window)
        dd = (cummax - window) / cummax
        max_dd[i] = np.max(dd)

    # 波动率比 (20d vs 60d)
    vol_ratio = np.ones(n)
    for i in range(60, n):
        r20 = np.diff(navs[i - 20:i + 1]) / navs[i - 20:i]
        r60 = np.diff(navs[i - 60:i + 1]) / navs[i - 60:i]
        v20 = np.std(r20) if len(r20) > 1 else 0
        v60 = np.std(r60) if len(r60) > 1 else 0
        vol_ratio[i] = v20 / v60 if v60 > 0 else 1.0

    # 连跌/连涨天数
    consec_dec = np.zeros(n, dtype=int)
    consec_rise = np.zeros(n, dtype=int)
    for i in range(1, n):
        if navs[i] < navs[i - 1]:
            consec_dec[i] = consec_dec[i - 1] + 1
        else:
            consec_dec[i] = 0
        if navs[i] > navs[i - 1]:
            consec_rise[i] = consec_rise[i - 1] + 1
        else:
            consec_rise[i] = 0

    # 累计跌幅
    recent_decline = np.zeros(n)
    for i in range(1, n):
        d = consec_dec[i]
        if d > 0 and i - d >= 0:
            start = navs[i - d]
            recent_decline[i] = (navs[i] - start) / start if start > 0 else 0

    # 高点回撤 (20d)
    pullback = np.zeros(n)
    for i in range(19, n):
        peak = np.max(navs[i - 19:i + 1])
        pullback[i] = (peak - navs[i]) / peak if peak > 0 else 0

    # 月度收益 (22d)
    monthly = np.zeros(n)
    for i in range(22, n):
        monthly[i] = (navs[i] - navs[i - 22]) / navs[i - 22] if navs[i - 22] > 0 else 0

    # 趋势强度
    trend = np.zeros(n)
    for i in range(199, n):
        if ma200[i] > 0:
            trend[i] = (navs[i] - ma200[i]) / ma200[i]

    above_ma200 = navs > ma200
    golden_cross = ma100 > ma200

    return {
        "dates": dates, "navs": navs,
        "ma200": ma200, "ma100": ma100, "ma50": ma50,
        "rsi": rsi, "sharpe": sharpe, "max_dd": max_dd,
        "vol_ratio": vol_ratio,
        "consec_dec": consec_dec, "consec_rise": consec_rise,
        "recent_decline": recent_decline, "pullback": pullback,
        "monthly": monthly, "trend": trend,
        "above_ma200": above_ma200, "golden_cross": golden_cross,
    }


# ── 规则匹配 ──────────────────────────────────────────────

def check_buy(i, ind, rules):
    """检查第 i 天是否触发任一买入规则。返回 (rule_name or None)。"""
    if not ind["above_ma200"][i] or np.isnan(ind["ma200"][i]):
        return None
    cd = ind["consec_dec"][i]
    rsi = ind["rsi"][i]
    sharpe = ind["sharpe"][i]
    max_dd = ind["max_dd"][i]
    vr = ind["vol_ratio"][i]
    pull = ind["pullback"][i]
    trend = ind["trend"][i]

    r = rules.get("P1", {})
    if r.get("enabled", True) and r["decline_min"] <= cd <= r["decline_max"] and rsi < r["rsi_max"]:
        return "P1"
    r = rules.get("P2", {})
    if r.get("enabled", True) and rsi < r["rsi_max"] and max_dd > r["max_dd_min"]:
        return "P2"
    r = rules.get("P3", {})
    if r.get("enabled", True) and r["decline_min"] <= cd <= r["decline_max"] and sharpe > r["sharpe_min"]:
        return "P3"
    r = rules.get("P4", {})
    if r.get("enabled", True) and rsi < r["rsi_max"] and vr < r["vol_ratio_max"]:
        return "P4"
    r = rules.get("P5", {})
    if r.get("enabled", True) and max_dd > r["max_dd_min"] and rsi < r["rsi_max"]:
        return "P5"
    r = rules.get("P6", {})
    if r.get("enabled", True) and sharpe > r["sharpe_min"] and r["decline_min"] <= cd <= r["decline_max"]:
        return "P6"
    r = rules.get("P7", {})
    if r.get("enabled", True) and vr < r["vol_ratio_max"] and pull > r["pullback_min"]:
        return "P7"
    r = rules.get("P8", {})
    if r.get("enabled", True) and trend > r["trend_min"] and cd >= r["decline_min"]:
        return "P8"
    return None


def check_sell(i, ind, rules):
    """检查第 i 天是否触发任一卖出规则。返回 (action, rule_name) or (None, None)。
    action: '清仓' or '减仓'。清仓优先于减仓。"""
    rsi = ind["rsi"][i]
    sharpe = ind["sharpe"][i]
    max_dd = ind["max_dd"][i]
    vr = ind["vol_ratio"][i]
    pull = ind["pullback"][i]
    trend = ind["trend"][i]
    monthly = ind["monthly"][i]
    cd = ind["consec_dec"][i]
    cr = ind["consec_rise"][i]
    above = ind["above_ma200"][i]

    # 清仓类规则优先
    r = rules.get("S1", {})
    if r.get("enabled", True) and max_dd > r["max_dd_min"]:
        return "清仓", "S1"
    r = rules.get("S8", {})
    if r.get("enabled", False) and not above and monthly < r["monthly_max"]:
        return "清仓", "S8"

    # 减仓类规则
    r = rules.get("S2", {})
    if r.get("enabled", r.get("enabled", False)) and cr >= r["rise_min"] and rsi > r["rsi_min"]:
        return "减仓", "S2"
    r = rules.get("S3", {})
    if r.get("enabled", True) and monthly > r["monthly_min"] and rsi > r["rsi_min"]:
        return "减仓", "S3"
    r = rules.get("S4", {})
    if r.get("enabled", True) and sharpe < r["sharpe_max"] and monthly < r["monthly_max"]:
        return "减仓", "S4"
    r = rules.get("S5", {})
    if r.get("enabled", True) and above and pull > r["pullback_min"] and cd >= r["decline_min"]:
        return "减仓", "S5"
    r = rules.get("S6", {})
    if r.get("enabled", True) and trend > r["trend_min"] and rsi > r["rsi_min"]:
        return "减仓", "S6"
    r = rules.get("S7", {})
    if r.get("enabled", True) and vr > r["vol_ratio_min"]:
        return "减仓", "S7"
    return None, None


# ── 单基金回测 ─────────────────────────────────────────────

def backtest_fund(records, start_date, end_date, buy_rules, sell_rules, swing_exit=None):
    """单基金 swing trading 回测。"""
    if swing_exit is None:
        swing_exit = SWING_EXIT
    ind = precompute(records)
    if ind is None:
        return None
    return backtest_fund_with_ind(ind, start_date, end_date, buy_rules, sell_rules, swing_exit)


def backtest_fund_with_ind(ind, start_date, end_date, buy_rules, sell_rules, swing_exit=None):
    """单基金回测（使用预计算指标，避免重复计算）。

    start_invested: True=起始满仓（牛市捕获涨幅，仅大跌退出），False=起始空仓（等买入信号）
    """
    if swing_exit is None:
        swing_exit = SWING_EXIT
    if ind is None:
        return None

    dates = ind["dates"]
    navs = ind["navs"]
    n = len(navs)

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

    # Buy & Hold（含买入手续费，与策略公平对比）
    nav0 = navs[start_idx]
    nav_end = navs[end_idx]
    if nav0 <= 0:
        return None
    bh_return = (1 - FEE_BUY) * nav_end / nav0 - 1

    # 策略回测
    start_invested = swing_exit.get("start_invested", False)
    cash = INITIAL_CAPITAL
    units = 0.0
    pos = 0.0
    trades = 0
    last_buy_day = -999
    last_sell_day = -999
    buy_nav = 0.0
    peak_nav = 0.0  # 买入后最高净值（用于移动止盈）
    trade_log = []

    SHORT_TERM_FEE = 0.015  # 持有<7天赎回费
    SHORT_TERM_DAYS = 7

    # 起始满仓模式：第一天直接买入
    first_nav = navs[start_idx]
    if start_invested and first_nav > 0:
        buy_amt = cash * 0.999  # 留少量缓冲防取整
        fee = buy_amt * FEE_BUY
        units += (buy_amt - fee) / first_nav
        cash -= buy_amt
        pos = units * first_nav / (cash + units * first_nav)
        last_buy_day = start_idx
        buy_nav = first_nav
        peak_nav = first_nav
        trade_log.append((dates[start_idx], "BUY", first_nav, "起始满仓"))

    for i in range(start_idx, end_idx + 1):
        nav = navs[i]
        if nav <= 0 or np.isnan(ind["ma200"][i]):
            continue

        tv = cash + units * nav

        if pos < 0.01:
            # 空仓 → 检查买入（卖出后需冷却）
            if i - last_sell_day < swing_exit.get("buy_cooldown", 0):
                continue
            # 重新入场模式：起始满仓模式下，退出后用不同信号重新入场
            reentry_mode = swing_exit.get("reentry_mode", "dip")
            reentry_max_dd = swing_exit.get("reentry_max_dd", 1.0)  # 重新入场需回撤低于此值
            rule = None
            # 先检查回撤恢复条件（防止S1反复触发）
            if ind["max_dd"][i] > reentry_max_dd:
                rule = None  # 回撤未恢复，不重新入场
            elif reentry_mode == "trend" and start_invested:
                # 趋势恢复：站上MA200 + MA100>MA200（金叉）
                if ind["above_ma200"][i] and ind["golden_cross"][i]:
                    rule = "趋势恢复"
            elif reentry_mode == "rsi50" and start_invested:
                # RSI回升到50以上（动量转正）
                if ind["rsi"][i] > 50 and ind["above_ma200"][i]:
                    rule = "RSI恢复"
            elif reentry_mode == "ma10" and start_invested:
                # 站上10日均线（快速趋势恢复）
                ma10 = np.mean(ind["navs"][max(0,i-9):i+1])
                if nav > ma10 and ind["above_ma200"][i]:
                    rule = "MA10恢复"
            elif reentry_mode == "ma20" and start_invested:
                # 站上20日均线
                ma20 = np.mean(ind["navs"][max(0,i-19):i+1])
                if nav > ma20 and ind["above_ma200"][i]:
                    rule = "MA20恢复"
            elif reentry_mode == "quick" and start_invested:
                # 快速恢复：RSI>40 + 站上10日均线
                ma10 = np.mean(ind["navs"][max(0,i-9):i+1])
                if ind["rsi"][i] > 40 and nav > ma10:
                    rule = "快速恢复"
            elif reentry_mode == "either" and start_invested:
                # 任一信号：P1-P8 抄底 或 站上MA200
                rule = check_buy(i, ind, buy_rules)
                if not rule and ind["above_ma200"][i] and ind["golden_cross"][i]:
                    rule = "趋势恢复"
            else:
                rule = check_buy(i, ind, buy_rules)
            if rule and cash > 1000:
                buy_amt = cash * 0.999  # 留少量缓冲防取整
                fee = buy_amt * FEE_BUY
                units += (buy_amt - fee) / nav
                cash -= buy_amt
                trades += 1
                pos = units * nav / tv
                last_buy_day = i
                buy_nav = nav
                peak_nav = nav
                trade_log.append((dates[i], "BUY", nav, rule))
        else:
            # 更新峰值
            if nav > peak_nav:
                peak_nav = nav

            # 持仓 → 检查退出
            days_held = i - last_buy_day
            pnl = (nav - buy_nav) / buy_nav if buy_nav > 0 else 0
            drawdown = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0
            min_hold = swing_exit.get("min_hold_days", 0)
            exit_reason = None

            # 1. 止损（不受 min_hold 限制）
            if pnl <= swing_exit.get("stop_loss", -1):
                exit_reason = f"止损({pnl:+.1%})"
            # 以下规则受 min_hold 限制
            elif days_held >= min_hold:
                # 2. 移动止盈（从峰值回撤）
                if drawdown >= swing_exit.get("trailing_stop", 1):
                    exit_reason = f"移动止盈({pnl:+.1%},DD{drawdown:.1%})"
                # 3. 超时清仓
                elif days_held >= swing_exit.get("max_hold_days", 999):
                    exit_reason = f"超时({days_held}d,{pnl:+.1%})"
                # 4. S1 硬止损（60日回撤）
                else:
                    action, srule = check_sell(i, ind, sell_rules)
                    if action == "清仓":
                        exit_reason = srule

            if exit_reason and days_held >= 1:
                # 短期赎回费：持有<7天用1.5%，否则0.5%
                sell_fee = SHORT_TERM_FEE if days_held < SHORT_TERM_DAYS else FEE_SELL
                gross = units * nav
                fee = gross * sell_fee
                cash += gross - fee
                units = 0.0
                trades += 1
                pos = 0.0
                last_sell_day = i
                trade_log.append((dates[i], "SELL", nav, exit_reason))

    final_value = cash + units * navs[end_idx]
    strat_return = final_value / INITIAL_CAPITAL - 1

    return {
        "strat_return": strat_return,
        "bh_return": bh_return,
        "excess": strat_return - bh_return,
        "trades": trades,
        "trade_log": trade_log,
    }


# ── 批量回测 ─────────────────────────────────────────────

def load_funds(n=None, include_benchmark=True):
    """加载基金列表和净值数据。n=None 加载全部 200 只。"""
    with open(SAMPLE, "r", encoding="utf-8") as f:
        funds = json.load(f)
    if n:
        funds = funds[:n]
    if include_benchmark:
        codes = [f["code"] for f in funds]
        if BENCHMARK not in codes:
            funds.append({"code": BENCHMARK, "name": "易方达沪深300ETF联接A", "type": "ETF联接", "aum_yi": 100})

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
                "records": [{"date": r[0], "nav": r[1]} for r in rows],
            }
    conn.close()
    return list(fund_data.values())


def run_batch(funds, buy_rules, sell_rules, windows=None, swing_exit=None, verbose=True):
    """批量回测所有基金 × 所有窗口。"""
    if windows is None:
        windows = WINDOWS

    # {window_name: {code: result}}
    results = {w[0]: {} for w in windows}

    for fi, fund in enumerate(funds):
        code = fund["code"]
        records = fund["records"]
        for wname, sdate, edate in windows:
            r = backtest_fund(records, sdate, edate, buy_rules, sell_rules, swing_exit)
            if r:
                results[wname][code] = r
        if verbose and (fi + 1) % 20 == 0:
            print(f"  进度: {fi+1}/{len(funds)}", flush=True)

    return results


def summarize(results, funds_meta=None):
    """汇总结果。"""
    meta = {f["code"]: f for f in (funds_meta or [])}
    print("\n" + "=" * 80)
    print("Swing Trading 回测结果汇总")
    print("=" * 80)

    for wname, code_results in results.items():
        if not code_results:
            continue
        n = len(code_results)
        wins = sum(1 for r in code_results.values() if r["excess"] > 0)
        avg_strat = np.mean([r["strat_return"] for r in code_results.values()])
        avg_bh = np.mean([r["bh_return"] for r in code_results.values()])
        avg_excess = np.mean([r["excess"] for r in code_results.values()])
        med_excess = np.median([r["excess"] for r in code_results.values()])
        avg_trades = np.mean([r["trades"] for r in code_results.values()])

        # 基准基金
        bm = code_results.get(BENCHMARK)

        print(f"\n--- {wname} ({n} 只基金) ---")
        print(f"  胜率:        {wins}/{n} = {wins/n:.1%}")
        print(f"  平均策略收益: {avg_strat:+.2%}")
        print(f"  平均持有收益: {avg_bh:+.2%}")
        print(f"  平均超额:    {avg_excess:+.2%}  (中位 {med_excess:+.2%})")
        print(f"  平均交易:    {avg_trades:.1f} 笔")
        if bm:
            print(f"  沪深300:     策略 {bm['strat_return']:+.2%} vs 持有 {bm['bh_return']:+.2%}  超额 {bm['excess']:+.2%}  ({bm['trades']}笔)")

    # 交叉窗口一致性
    all_codes = set()
    for cr in results.values():
        all_codes.update(cr.keys())
    if len(results) >= 2:
        wnames = list(results.keys())
        all_win = 0
        all_lose = 0
        for code in all_codes:
            excesses = [results[wn].get(code, {}).get("excess", None) for wn in wnames]
            excesses = [e for e in excesses if e is not None]
            if len(excesses) == len(wnames):
                if all(e > 0 for e in excesses):
                    all_win += 1
                elif all(e <= 0 for e in excesses):
                    all_lose += 1
        n_valid = len([c for c in all_codes if all(wn in results and c in results[wn] for wn in wnames)])
        print(f"\n--- 交叉窗口一致性 ({len(wnames)} 窗口) ---")
        print(f"  全部跑赢: {all_win}/{n_valid} = {all_win/n_valid:.1%}" if n_valid else "  无有效数据")
        print(f"  全部跑输: {all_lose}/{n_valid} = {all_lose/n_valid:.1%}" if n_valid else "")


# ── MA200 非对称确认回测 ─────────────────────────────────

def backtest_ma200_asym(ind, start_date, end_date, exit_params=None):
    """MA200 非对称确认择时回测。

    退出：跌破 MA200 连续 exit_confirm_days 天 → 清仓
    重新入场：站上 MA200 连续 entry_confirm_days 天 → 满仓买回
    起始满仓。
    """
    if exit_params is None:
        exit_params = SWING_EXIT
    if ind is None:
        return None

    dates = ind["dates"]
    navs = ind["navs"]
    n = len(navs)

    ma_p = exit_params.get("ma_period", 200)
    exit_cd = exit_params.get("exit_confirm_days", 12)
    entry_cd = exit_params.get("entry_confirm_days", 1)
    buf = exit_params.get("buffer", 0.0)
    cooldown = exit_params.get("buy_cooldown", 3)
    min_hold = exit_params.get("min_hold_days", 5)

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

    # Buy & Hold（含买入手续费，公平对比）
    nav0 = navs[start_idx]
    nav_end = navs[end_idx]
    if nav0 <= 0:
        return None
    bh_return = (1 - FEE_BUY) * nav_end / nav0 - 1

    ma = ind[f"ma{ma_p}"]
    cash = INITIAL_CAPITAL
    units = 0.0
    trades = 0
    last_sell = -999
    last_buy = -999
    below_count = 0
    above_count = 0
    trade_log = []

    # 起始满仓
    if nav0 > 0:
        buy_amt = cash * 0.999
        fee = buy_amt * FEE_BUY
        units = (buy_amt - fee) / nav0
        cash -= buy_amt
        last_buy = start_idx
        trade_log.append((dates[start_idx], "BUY", nav0, "起始满仓"))

    for i in range(start_idx, end_idx + 1):
        nav = navs[i]
        if nav <= 0 or np.isnan(ma[i]):
            continue

        # 确认计数
        if nav < ma[i] * (1 - buf):
            below_count += 1
            above_count = 0
        elif nav > ma[i] * (1 + buf):
            above_count += 1
            below_count = 0
        else:
            below_count = 0
            above_count = 0

        if units < 0.01:
            # 空仓 → 检查重新入场
            if i - last_sell < cooldown:
                continue
            if above_count >= entry_cd and cash > 1000:
                buy_amt = cash * 0.999
                fee = buy_amt * FEE_BUY
                units += (buy_amt - fee) / nav
                cash -= buy_amt
                trades += 1
                last_buy = i
                trade_log.append((dates[i], "BUY", nav, f"站上MA{ma_p}×{entry_cd}d"))
        else:
            # 持仓 → 检查退出
            if i - last_buy >= min_hold and below_count >= exit_cd:
                gross = units * nav
                fee = gross * FEE_SELL
                cash += gross - fee
                units = 0.0
                trades += 1
                last_sell = i
                trade_log.append((dates[i], "SELL", nav, f"跌破MA{ma_p}×{exit_cd}d"))

    final_value = cash + units * navs[end_idx]
    strat_return = final_value / INITIAL_CAPITAL - 1

    return {
        "strat_return": strat_return,
        "bh_return": bh_return,
        "excess": strat_return - bh_return,
        "trades": trades,
        "trade_log": trade_log,
    }


def run_batch_ma200(funds, exit_params=None, windows=None, verbose=True):
    """批量 MA200 非对称确认回测。"""
    if windows is None:
        windows = WINDOWS
    results = {w[0]: {} for w in windows}

    # 预计算指标
    precomputed = {}
    for fund in funds:
        ind = precompute(fund["records"])
        if ind is not None:
            precomputed[fund["code"]] = ind

    for fi, (code, ind) in enumerate(precomputed.items()):
        for wname, sdate, edate in windows:
            r = backtest_ma200_asym(ind, sdate, edate, exit_params)
            if r:
                results[wname][code] = r
        if verbose and (fi + 1) % 20 == 0:
            print(f"  进度: {fi+1}/{len(precomputed)}", flush=True)

    return results, precomputed


# ── 主入口 ─────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Swing trading 回测（MA200 非对称确认择时）")
    parser.add_argument("--n", type=int, default=None, help="基金数量（默认全部200+基准）")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("加载基金数据...", flush=True)
    funds = load_funds(n=args.n, include_benchmark=True)
    print(f"共 {len(funds)} 只基金（含沪深300基准）", flush=True)

    print(f"\n策略: MA200 非对称确认择时（退出{SWING_EXIT['exit_confirm_days']}d + 入场{SWING_EXIT['entry_confirm_days']}d）")
    print(f"运行回测（{len(WINDOWS)} 窗口）...", flush=True)
    results, precomputed = run_batch_ma200(funds, SWING_EXIT, verbose=not args.quiet)
    summarize(results, funds)

    # 多窗口一致性分析
    wnames = [w[0] for w in WINDOWS]
    fund_wins = defaultdict(list)
    for wn in wnames:
        for code, r in results[wn].items():
            fund_wins[code].append(r["excess"] > 0)

    # 只统计有全部窗口数据的基金
    full_funds = {c: w for c, w in fund_wins.items() if len(w) == len(wnames)}
    if full_funds:
        win_ge2 = sum(1 for w in full_funds.values() if sum(w) >= 2)
        win_ge3 = sum(1 for w in full_funds.values() if sum(w) >= 3)
        total = len(full_funds)
        bm = full_funds.get(BENCHMARK, [])
        print(f"\n--- 多窗口一致性（{len(wnames)} 窗口，{total} 只基金有完整数据）---")
        print(f"  ≥2窗口赢: {win_ge2}/{total} = {win_ge2/total:.1%}")
        if len(wnames) >= 3:
            print(f"  ≥3窗口赢: {win_ge3}/{total} = {win_ge3/total:.1%}")
        print(f"  沪深300:  {sum(bm)}/{len(bm)} 窗口赢 → {bm}")

    # 保存明细
    out = "/mnt/d/software/USTC-Drive/科大云盘/charming-cloud/project/quant-fund/swing_results.json"
    serializable = {}
    for wn, cr in results.items():
        serializable[wn] = {
            code: {k: v for k, v in r.items() if k != "trade_log"}
            for code, r in cr.items()
        }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n明细已保存: {out}")


if __name__ == "__main__":
    main()
