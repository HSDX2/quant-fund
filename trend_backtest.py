#!/usr/bin/env python3
"""趋势跟踪策略回测：在沪深300上测试多种学术策略，对比 buy-and-hold。

回测区间: 2021-07-01 ~ 2026-06-18 (约5年，覆盖2021-2022下跌和2024-2025反弹)
基金: 110020 易方达沪深300ETF联接A (数据从2009年开始，足够MA200预热)
"""

import sqlite3
import numpy as np
from datetime import datetime, timedelta

DB_PATH = "/mnt/d/software/USTC-Drive/科大云盘/charming-cloud/project/quant-fund/cache/nav_cache.db"
CODE = "110020"
START_DATE = "2023-06-18"
END_DATE = "2026-06-18"
INITIAL_CAPITAL = 100000.0
FEE_BUY = 0.0015
FEE_SELL = 0.005

# ── 数据加载 ──────────────────────────────────────────

def load_nav_data():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT date, nav FROM nav_history WHERE code = ? ORDER BY date",
        (CODE,),
    ).fetchall()
    conn.close()
    return [{"date": r[0], "nav": r[1]} for r in rows]


def get_start_idx(all_records):
    """找到回测起始索引（需要200天历史做MA200）"""
    start_idx = None
    for i, r in enumerate(all_records):
        if r["date"] >= START_DATE:
            start_idx = i
            break
    if start_idx is None or start_idx < 200:
        start_idx = 200
    return start_idx


# ── 通用回测引擎 ──────────────────────────────────────

def backtest(all_records, start_idx, strategy_fn, strategy_name):
    """通用回测引擎。
    
    Args:
      all_records: 完整净值数据（含预热期）
      start_idx: 回测起始索引（all_records[start_idx] 是回测第一天）
      strategy_fn: (records_up_to_today, current_position) -> (target_weight, reason)
    
    交易逻辑:
      - 目标仓位 > 当前仓位 → 买入差额（扣手续费）
      - 目标仓位 < 当前仓位 → 卖出差额（扣手续费）
      - 仓位变化 < 5% 不操作（减少噪音交易）
    """
    bt_records = all_records[start_idx:]
    nav0 = bt_records[0]["nav"]
    
    # buy-and-hold 基准（从回测第一天买入持有到最后）
    buyhold_units = (INITIAL_CAPITAL * (1 - FEE_BUY)) / nav0
    buyhold_final = buyhold_units * bt_records[-1]["nav"]
    buyhold_return = buyhold_final / INITIAL_CAPITAL - 1
    
    # 策略
    cash = INITIAL_CAPITAL
    units = 0.0
    position = 0.0  # 当前仓位比例
    trades = 0
    trade_log = []
    
    for i in range(len(bt_records)):
        today = bt_records[i]
        nav = today["nav"]
        if nav <= 0:
            continue
        
        # 截至当日的完整数据（含预热期）
        data_so_far = all_records[:start_idx + i + 1]
        if len(data_so_far) < 200:
            continue
        
        # 策略决策
        target_weight, reason = strategy_fn(data_so_far, position)
        target_weight = max(0.0, min(1.0, target_weight))
        
        # 计算当前总资产
        total_value = cash + units * nav
        
        # 仓位调整
        target_value = total_value * target_weight
        current_invested = units * nav
        diff = target_value - current_invested
        
        if abs(diff) < total_value * 0.05:  # 变化<5%不操作
            continue
        
        if diff > 0 and cash > 100:
            # 买入
            buy_amount = min(diff, cash * 0.98)
            fee = buy_amount * FEE_BUY
            buy_units = (buy_amount - fee) / nav
            cash -= buy_amount
            units += buy_units
            trades += 1
            new_pos = units * nav / (cash + units * nav)
            if reason:
                trade_log.append({
                    "date": today["date"], "action": "BUY",
                    "weight": new_pos, "nav": nav, "reason": reason
                })
            position = new_pos
            
        elif diff < 0 and units > 0:
            # 卖出
            sell_units = min(units * abs(diff) / current_invested, units)
            if sell_units > 0:
                gross = sell_units * nav
                fee = gross * FEE_SELL
                cash += gross - fee
                units -= sell_units
                trades += 1
                new_pos = units * nav / (cash + units * nav) if (cash + units * nav) > 0 else 0
                if reason:
                    trade_log.append({
                        "date": today["date"], "action": "SELL",
                        "weight": new_pos, "nav": nav, "reason": reason
                    })
                position = new_pos
    
    final_value = cash + units * bt_records[-1]["nav"]
    strategy_return = final_value / INITIAL_CAPITAL - 1
    
    return {
        "name": strategy_name,
        "strategy_return": strategy_return,
        "buyhold_return": buyhold_return,
        "excess": strategy_return - buyhold_return,
        "final_value": final_value,
        "buyhold_value": buyhold_final,
        "trades": trades,
        "trade_log": trade_log,
    }


# ── 辅助函数 ──────────────────────────────────────────

def sma(data, period):
    """简单移动平均"""
    if len(data) < period:
        return None
    return np.mean([r["nav"] for r in data[-period:]])

def ema(data, period):
    """指数移动平均"""
    navs = [r["nav"] for r in data]
    if len(navs) < period:
        return None
    k = 2 / (period + 1)
    ema_val = navs[0]
    for nav in navs[1:]:
        ema_val = nav * k + ema_val * (1 - k)
    return ema_val

def realized_vol(data, window=20):
    """已实现波动率（年化）"""
    navs = [r["nav"] for r in data[-(window+1):]]
    if len(navs) < 2:
        return 0.0
    returns = np.diff(navs) / navs[:-1]
    return np.std(returns) * np.sqrt(244)

def momentum(data, lookback):
    """过去lookback天的收益率"""
    navs = [r["nav"] for r in data]
    if len(navs) < lookback + 1:
        return 0.0
    return navs[-1] / navs[-(lookback+1)] - 1


# ── 策略实现 ──────────────────────────────────────────

def strategy_buyhold(data, pos):
    """基准：买入持有"""
    return 1.0, None

def strategy_ma200(data, pos):
    """规则1: Faber MA200 择时 (Faber 2007)
    价格 > MA200 → 满仓; 价格 < MA200 → 清仓
    """
    ma = sma(data, 200)
    if ma is None:
        return pos, None
    if data[-1]["nav"] > ma:
        return 1.0, "Above MA200"
    else:
        return 0.0, "Below MA200"

def strategy_ma200_buffer(data, pos):
    """规则1b: MA200 + 1%缓冲带 (减少whipsaw)
    价格 > MA200*1.01 → 满仓; 价格 < MA200*0.99 → 清仓; 否则维持
    """
    ma = sma(data, 200)
    if ma is None:
        return pos, None
    nav = data[-1]["nav"]
    if nav > ma * 1.01:
        return 1.0, "Above MA200+1%"
    elif nav < ma * 0.99:
        return 0.0, "Below MA200-1%"
    return pos, None

def strategy_ma50_ma200(data, pos):
    """规则3: MA50/MA200 交叉 (Brock et al. 1992)
    MA50 > MA200 → 满仓; MA50 < MA200 → 清仓
    """
    ma50 = sma(data, 50)
    ma200 = sma(data, 200)
    if ma50 is None or ma200 is None:
        return pos, None
    if ma50 > ma200:
        return 1.0, "MA50>MA200 golden cross"
    else:
        return 0.0, "MA50<MA200 death cross"

def strategy_tsmom_252(data, pos):
    """规则4: 时间序列动量 (Moskowitz et al. 2012)
    过去252天收益 > 0 → 满仓; < 0 → 清仓
    """
    mom = momentum(data, 252)
    if mom > 0:
        return 1.0, f"TSMOM +{mom:.1%}"
    else:
        return 0.0, f"TSMOM {mom:.1%}"

def strategy_tsmom_126(data, pos):
    """规则4b: TSMOM 半年lookback"""
    mom = momentum(data, 126)
    if mom > 0:
        return 1.0, f"TSMOM6m +{mom:.1%}"
    else:
        return 0.0, f"TSMOM6m {mom:.1%}"

def strategy_vol_target(data, pos):
    """规则5: 波动率目标 (Moreira & Muir 2017)
    仓位 = target_vol / realized_vol，限制0-100%
    """
    rv = realized_vol(data, 20)
    if rv <= 0:
        return pos, None
    target_vol = 0.15  # 15%年化目标
    weight = min(target_vol / rv, 1.0)
    return weight, f"Vol target {rv:.1%}→{weight:.0%}"

def strategy_ma200_vol(data, pos):
    """规则7: MA200 + 波动率目标组合
    价格>MA200时按vol target调仓; 价格<MA200清仓
    """
    ma = sma(data, 200)
    if ma is None:
        return pos, None
    if data[-1]["nav"] < ma:
        return 0.0, "Below MA200"
    rv = realized_vol(data, 20)
    if rv <= 0:
        return 1.0, "Above MA200, vol=0"
    target_vol = 0.15
    weight = min(target_vol / rv, 1.0)
    return weight, f"MA200+VolT {weight:.0%}"

def strategy_dual_momentum(data, pos):
    """规则8: 12-1动量 + MA200过滤
    过去252天(去掉最近21天)涨幅>0 且 价格>MA200 → 满仓
    否则清仓
    """
    ma = sma(data, 200)
    if ma is None:
        return pos, None
    navs = [r["nav"] for r in data]
    if len(navs) < 253:
        return pos, None
    mom_12_1 = navs[-22] / navs[-253] - 1  # 12个月前到1个月前
    above_ma = navs[-1] > ma
    if mom_12_1 > 0 and above_ma:
        return 1.0, f"12-1 mom +{mom_12_1:.1%} & above MA200"
    else:
        return 0.0, f"12-1 mom {mom_12_1:.1%} or below MA200"

def strategy_ma200_dd(data, pos):
    """规则6+1: MA200 + 回撤控制
    价格>MA200 且 从高点回撤<10% → 满仓
    价格<MA200 或 回撤>15% → 清仓
    回撤10-15% → 半仓
    """
    ma = sma(data, 200)
    if ma is None:
        return pos, None
    nav = data[-1]["nav"]
    navs = [r["nav"] for r in data]
    high_water = max(navs[-252:]) if len(navs) >= 252 else max(navs)
    drawdown = (high_water - nav) / high_water if high_water > 0 else 0
    
    if nav < ma:
        return 0.0, "Below MA200"
    if drawdown > 0.15:
        return 0.0, f"DD {drawdown:.1%}>15%"
    if drawdown > 0.10:
        return 0.5, f"DD {drawdown:.1%}, half position"
    return 1.0, f"Above MA200, DD {drawdown:.1%}"

def strategy_donchian_55(data, pos):
    """规则9: Donchian 55日通道突破
    价格创55日新高 → 满仓; 创55日新低 → 清仓
    """
    navs = [r["nav"] for r in data]
    if len(navs) < 56:
        return pos, None
    upper = max(navs[-56:-1])  # 过去55日最高（不含今日）
    lower = min(navs[-56:-1])
    nav = navs[-1]
    if nav > upper:
        return 1.0, f"Breakout 55d high"
    elif nav < lower:
        return 0.0, f"Breakdown 55d low"
    return pos, None

def strategy_macd_ma200(data, pos):
    """规则10: MACD + MA200
    MACD金叉 且 价格>MA200 → 满仓
    MACD死叉 或 价格<MA200 → 清仓
    """
    ma = sma(data, 200)
    if ma is None:
        return pos, None
    nav = data[-1]["nav"]
    if nav < ma:
        return 0.0, "Below MA200"
    
    navs = [r["nav"] for r in data]
    if len(navs) < 35:
        return pos, None
    
    ema12 = ema(data, 12)
    ema26 = ema(data, 26)
    if ema12 is None or ema26 is None:
        return pos, None
    macd = ema12 - ema26
    
    # 简化：MACD > 0 且 价格>MA200 → 持有
    if macd > 0 and nav > ma:
        return 1.0, f"MACD>0 & above MA200"
    else:
        return 0.0, f"MACD<0 or below MA200"


# ── 迭代优化策略 ──────────────────────────────────────

def _is_month_end(data):
    """判断今天是否是月末（当月最后一个交易日）"""
    if len(data) < 2:
        return False
    today_month = data[-1]["date"][:7]
    tomorrow_month = data[-2]["date"][:7] if len(data) >= 2 else today_month
    return today_month != tomorrow_month or len(data) == 1

def strategy_tsmom_monthly(data, pos):
    """TSMOM 月度调仓（减少交易次数和手续费）
    仅在月末检查信号，避免日内噪音。
    """
    if not _is_month_end(data) and pos > 0:
        return pos, None  # 非月末，维持仓位
    mom = momentum(data, 252)
    if mom > 0:
        return 1.0, f"Monthly TSMOM +{mom:.1%}"
    else:
        return 0.0, f"Monthly TSMOM {mom:.1%}"

def strategy_tsmom_ma_cross(data, pos):
    """TSMOM + MA50/200交叉双重确认
    两个信号都看涨才满仓，都看跌才清仓，不一致则维持。
    """
    ma50 = sma(data, 50)
    ma200 = sma(data, 200)
    if ma50 is None or ma200 is None:
        return pos, None
    mom = momentum(data, 252)
    
    tsmom_bull = mom > 0
    cross_bull = ma50 > ma200
    
    if tsmom_bull and cross_bull:
        return 1.0, f"Both bull: TSMOM+{mom:.1%}, golden cross"
    elif not tsmom_bull and not cross_bull:
        return 0.0, f"Both bear: TSMOM{mom:.1%}, death cross"
    return pos, f"Mixed: TSMOM{mom:.1%}, {'golden' if cross_bull else 'death'}"

def strategy_ma_cross_monthly(data, pos):
    """MA50/200交叉 月度调仓
    仅月末检查，减少whipsaw。
    """
    if not _is_month_end(data) and pos > 0:
        return pos, None
    ma50 = sma(data, 50)
    ma200 = sma(data, 200)
    if ma50 is None or ma200 is None:
        return pos, None
    if ma50 > ma200:
        return 1.0, "Monthly golden cross"
    else:
        return 0.0, "Monthly death cross"

def strategy_triple_confirm(data, pos):
    """三重确认：TSMOM + MA200 + MA50/200交叉
    三个信号至少两个看涨才满仓，至少两个看跌才清仓。
    """
    ma50 = sma(data, 50)
    ma200 = sma(data, 200)
    if ma50 is None or ma200 is None:
        return pos, None
    mom = momentum(data, 252)
    
    signals_bull = sum([
        mom > 0,           # TSMOM
        data[-1]["nav"] > ma200,  # 价格在MA200上方
        ma50 > ma200,      # 金叉
    ])
    
    if signals_bull >= 2:
        return 1.0, f"Triple {signals_bull}/3 bull (mom {mom:+.1%})"
    elif signals_bull <= 1:
        return 0.0, f"Triple {signals_bull}/3 bull (mom {mom:+.1%})"
    return pos, f"Triple {signals_bull}/3 bull"

def strategy_tsmom_vol_filter(data, pos):
    """TSMOM + 波动率过滤
    动量为正且波动率低于历史均值 → 满仓
    动量为正但波动率飙升 → 半仓
    动量为负 → 清仓
    """
    mom = momentum(data, 252)
    if mom <= 0:
        return 0.0, f"TSMOM {mom:.1%}, exit"
    
    rv = realized_vol(data, 20)
    hv = realized_vol(data, 60)
    if hv <= 0:
        return 1.0, f"TSMOM +{mom:.1%}, vol stable"
    
    if rv <= hv:
        return 1.0, f"TSMOM +{mom:.1%}, vol {rv:.1%}<{hv:.1%}"
    else:
        return 0.5, f"TSMOM +{mom:.1%}, vol spike {rv:.1%}>{hv:.1%}, half"

def strategy_tsmom_ma200_combo(data, pos):
    """TSMOM 252d + MA200 组合
    动量为正 且 价格>MA200 → 满仓
    动量为正 但 价格<MA200 → 半仓（等待趋势确认）
    动量为负 → 清仓
    """
    ma200 = sma(data, 200)
    if ma200 is None:
        return pos, None
    mom = momentum(data, 252)
    above_ma = data[-1]["nav"] > ma200
    
    if mom > 0 and above_ma:
        return 1.0, f"TSMOM+{mom:.1%} & above MA200"
    elif mom > 0 and not above_ma:
        return 0.5, f"TSMOM+{mom:.1%} but below MA200, half"
    else:
        return 0.0, f"TSMOM{mom:.1%}, exit"


# ── 主流程 ────────────────────────────────────────────

def main():
    all_records = load_nav_data()
    print(f"基金: {CODE}")
    print(f"数据范围: {all_records[0]['date']} ~ {all_records[-1]['date']}, {len(all_records)} 条")
    
    start_idx = get_start_idx(all_records)
    bt_records = all_records[start_idx:]
    print(f"回测区间: {bt_records[0]['date']} ~ {bt_records[-1]['date']}, {len(bt_records)} 条")
    print(f"初始资金: ¥{INITIAL_CAPITAL:,.0f}")
    print(f"手续费: 买入{FEE_BUY:.2%}, 卖出{FEE_SELL:.2%}")
    print()
    
    strategies = [
        ("Buy & Hold", strategy_buyhold),
        # 第一轮：基础策略
        ("MA200 (Faber 2007)", strategy_ma200),
        ("MA50/MA200 Cross", strategy_ma50_ma200),
        ("TSMOM 252d (Moskowitz)", strategy_tsmom_252),
        ("12-1 Momentum + MA200", strategy_dual_momentum),
        # 第二轮：迭代优化
        ("TSMOM Monthly", strategy_tsmom_monthly),
        ("TSMOM + MA Cross", strategy_tsmom_ma_cross),
        ("MA Cross Monthly", strategy_ma_cross_monthly),
        ("Triple Confirm", strategy_triple_confirm),
        ("TSMOM + Vol Filter", strategy_tsmom_vol_filter),
        ("TSMOM + MA200 Combo", strategy_tsmom_ma200_combo),
    ]
    
    results = []
    for name, fn in strategies:
        r = backtest(all_records, start_idx, fn, name)
        results.append(r)
    
    # 打印结果表
    print("=" * 90)
    print(f"  {'策略':<30} {'策略收益':>10} {'持有收益':>10} {'超额':>10} {'交易':>5} {'结论':>6}")
    print("=" * 90)
    
    for r in results:
        verdict = "✓ 赢" if r["excess"] > 0 else "✗ 输"
        print(
            f"  {r['name']:<30} "
            f"{r['strategy_return']:>+9.2%} "
            f"{r['buyhold_return']:>+9.2%} "
            f"{r['excess']:>+9.2%} "
            f"{r['trades']:>5} "
            f"{verdict:>6}"
        )
    print("=" * 90)
    
    # 打印赢家策略的交易明细
    winners = [r for r in results if r["excess"] > 0 and r["name"] != "Buy & Hold"]
    if winners:
        print(f"\n赢家策略 ({len(winners)} 个):")
        for w in winners:
            print(f"\n--- {w['name']} ---")
            print(f"  策略收益: {w['strategy_return']:+.2%} (¥{w['final_value']:,.0f})")
            print(f"  持有收益: {w['buyhold_return']:+.2%} (¥{w['buyhold_value']:,.0f})")
            print(f"  超额收益: {w['excess']:+.2%}")
            print(f"  交易次数: {w['trades']}")
            if w["trade_log"]:
                print(f"  交易明细 (前10笔):")
                for t in w["trade_log"][:10]:
                    print(f"    {t['date']} {t['action']:>4} → {t['weight']:.0%} @ {t['nav']:.4f}  {t['reason']}")
    else:
        print("\n无赢家策略")


if __name__ == "__main__":
    main()
