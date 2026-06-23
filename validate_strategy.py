#!/usr/bin/env python3
"""单基金策略有效性验证：对比"遵循建议"vs"持有不动"的收益。

方法:
  1. 从基金池随机选 100 只基金（含沪深300）
  2. 对每只基金，初始 100% 仓位：
     - 策略组：按报告的加仓/减仓比例操作（禁用清仓）
     - 基准组：持有不动
  3. 比较最终收益
"""

import argparse
import random
import sys
import os
import sqlite3
import json
import yaml
import numpy as np
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from indicators import compute_indicators
from holdings import analyze_holdings
from position_advisor import compute_position_advice

# ── 命令行参数 ────────────────────────────────────────

parser = argparse.ArgumentParser(description="单基金策略有效性验证")
parser.add_argument("--old", action="store_true", help="使用老基金列表（2015年前成立）+ 长回测区间")
parser.add_argument("--start", default=None, help="回测起始日期 (YYYY-MM-DD)")
parser.add_argument("--end", default=None, help="回测结束日期 (YYYY-MM-DD)")
args = parser.parse_args()

# ── 配置 ──────────────────────────────────────────────

with open(os.path.join(PROJECT_DIR, "config.yaml"), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

strategy = config["strategy"]
rules_cfg = strategy.get("rules")
target_vol = strategy.get("target_annual_vol", 0.40)

FEE_BUY = 0.0015
FEE_SELL = 0.005
FEE_SHORT = 0.015
SHORT_TERM_DAYS = 7

if args.old:
    START_DATE = args.start or "2016-01-01"
    END_DATE = args.end or "2026-06-18"
else:
    START_DATE = args.start or "2022-07-01"
    END_DATE = args.end or "2026-06-18"
INITIAL_CAPITAL = 100000.0

DB_PATH = os.path.join(PROJECT_DIR, "cache", "nav_cache.db")


# ── 数据加载 ──────────────────────────────────────────

def load_universe():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT value_json FROM kv_cache WHERE key = 'filtered_universe'"
    ).fetchone()
    conn.close()
    return json.loads(row[0])


def load_nav_data(conn, code):
    rows = conn.execute(
        "SELECT date, nav FROM nav_history WHERE code = ? ORDER BY date",
        (code,),
    ).fetchall()
    return [{"date": r[0], "nav": r[1]} for r in rows]


# ── 单基金模拟 ────────────────────────────────────────

def simulate_fund(code, name, fund_type, all_records, conn):
    """对单只基金模拟策略 vs 持有不动。

    Returns: dict or None
    """
    if len(all_records) < 250:
        return None

    # 找到回测起始点（需要 200 天历史做 MA200）
    start_idx = None
    for i, r in enumerate(all_records):
        if r["date"] >= START_DATE:
            start_idx = i
            break
    if start_idx is None or start_idx < 200:
        start_idx = 200
    if start_idx >= len(all_records):
        return None

    bt_records = all_records[start_idx:]
    if len(bt_records) < 30:
        return None

    nav0 = bt_records[0]["nav"]
    if nav0 <= 0:
        return None
    final_nav = bt_records[-1]["nav"]

    # === 持有不动 ===
    initial_fee = INITIAL_CAPITAL * FEE_BUY
    initial_units = (INITIAL_CAPITAL - initial_fee) / nav0
    buyhold_value = initial_units * final_nav
    buyhold_return = buyhold_value / INITIAL_CAPITAL - 1

    # === 策略：遵循加减仓建议 ===
    units = initial_units
    cash = 0.0
    trade_count = 0
    buy_date_str = bt_records[0]["date"]
    buy_date_dt = datetime.strptime(buy_date_str, "%Y-%m-%d")

    equity_types = ("股票", "混合", "指数", "QDII", "ETF", "LOF", "联接")
    is_equity = any(t in fund_type for t in equity_types)
    equity_codes = {code} if is_equity else set()

    for i in range(1, len(bt_records)):
        today_str = bt_records[i]["date"]
        current_nav = bt_records[i]["nav"]
        if current_nav <= 0:
            continue

        end_idx = start_idx + i + 1
        truncated = all_records[:end_idx]
        if len(truncated) < 200:
            continue

        # 计算指标
        try:
            indicators = compute_indicators(
                {code: truncated}, strategy, equity_codes, {code},
            )
        except Exception:
            continue
        if code not in indicators:
            continue

        # 获取持仓建议
        try:
            holdings_advice = analyze_holdings(
                [code], indicators, strategy, rules=rules_cfg,
            )
            _, hold_positions = compute_position_advice(
                indicators, {code: truncated}, [], [],
                holdings_advice, target_vol=target_vol,
            )
        except Exception:
            continue

        if not hold_positions:
            continue

        item = hold_positions[0]
        action = item.get("action", "持有")
        pct = item.get("pct", 0)

        if action == "清仓":
            continue  # 禁用清仓

        elif action == "减仓" and pct > 0:
            sell_units = units * pct
            if sell_units <= 0:
                continue
            gross = sell_units * current_nav
            hold_days = (
                datetime.strptime(today_str, "%Y-%m-%d") - buy_date_dt
            ).days
            fee_rate = FEE_SHORT if hold_days < SHORT_TERM_DAYS else FEE_SELL
            fee = gross * fee_rate
            cash += gross - fee
            units -= sell_units
            trade_count += 1

        elif action == "加仓" and pct > 0 and cash > 100:
            total_value = cash + units * current_nav
            amount = min(total_value * pct, cash * 0.95)
            if amount > 100:
                fee = amount * FEE_BUY
                buy_units = (amount - fee) / current_nav
                cash -= amount
                units += buy_units
                trade_count += 1

    strategy_value = cash + units * final_nav
    strategy_return = strategy_value / INITIAL_CAPITAL - 1

    return {
        "code": code,
        "name": name,
        "type": fund_type,
        "strategy_return": strategy_return,
        "buyhold_return": buyhold_return,
        "excess": strategy_return - buyhold_return,
        "strategy_value": strategy_value,
        "buyhold_value": buyhold_value,
        "trades": trade_count,
        "is_hs300": False,  # 后续标记
    }


# ── 主流程 ────────────────────────────────────────────

def main():
    random.seed(42)

    if args.old:
        # 老基金模式：从 old_funds.json 读取
        with open(os.path.join(PROJECT_DIR, "old_funds.json"), "r", encoding="utf-8") as f:
            selected = json.load(f)
        print(f"老基金模式: {len(selected)} 只（2015年前成立）")
        hs300_code = selected[0]["code"] if selected else "110020"
    else:
        universe = load_universe()
        print(f"基金池: {len(universe)} 只")

        # 找沪深300基金
        hs300_code = None
        hs300_keywords = ["沪深300", "沪深 300", "300ETF联接"]
        for f in universe:
            fname = f.get("name", "")
            if any(kw in fname for kw in hs300_keywords):
                hs300_code = f["code"]
                print(f"沪深300基金: {f['code']} {f['name']}")
                break

        if not hs300_code:
            # 005918 可能不在基金池中（被筛选排除），手动添加
            hs300_code = "005918"
            print(f"沪深300基金不在池中，手动添加: {hs300_code}")

        # 随机选 100 只（含沪深300）
        other_funds = [f for f in universe if f["code"] != hs300_code]
        selected = random.sample(other_funds, min(99, len(other_funds)))

        # 确保沪深300在列表中
        hs300_fund = next((f for f in universe if f["code"] == hs300_code), None)
        if hs300_fund is None:
            # 不在基金池中，手动构建
            hs300_fund = {"code": hs300_code, "name": "天弘沪深300ETF联接C", "type": "ETF联接"}
        selected.insert(0, hs300_fund)

    print(f"回测区间: {START_DATE} ~ {END_DATE}")
    print()

    # 模拟每只基金
    conn = sqlite3.connect(DB_PATH, timeout=10)
    results = []

    for idx, fund in enumerate(selected):
        code = fund["code"]
        name = fund.get("name", code)
        fund_type = fund.get("type", "混合型")

        all_records = load_nav_data(conn, code)
        if not all_records:
            continue

        result = simulate_fund(code, name, fund_type, all_records, conn)
        if result is None:
            continue

        result["is_hs300"] = (code == hs300_code)
        results.append(result)

        if (idx + 1) % 20 == 0:
            print(f"  进度: {idx + 1}/{len(selected)}")

    conn.close()

    if not results:
        print("无有效结果")
        return

    # ── 统计报告 ──────────────────────────────────────

    wins = [r for r in results if r["excess"] > 0]
    losses = [r for r in results if r["excess"] <= 0]

    print(f"\n{'=' * 72}")
    print(f"  单基金策略有效性验证报告")
    print(f"{'=' * 72}")
    print(f"  回测区间: {START_DATE} ~ {END_DATE}")
    print(f"  有效基金: {len(results)} 只")
    print()

    print(f"  策略优于持有不动: {len(wins)} 只 ({len(wins)/len(results)*100:.1f}%)")
    print(f"  策略劣于持有不动: {len(losses)} 只 ({len(losses)/len(results)*100:.1f}%)")
    print()

    # 沪深300
    hs300 = [r for r in results if r["is_hs300"]]
    if hs300:
        r = hs300[0]
        print(f"{'─' * 72}")
        print(f"  ★ 沪深300 基金")
        print(f"{'─' * 72}")
        print(f"  基金:     {r['code']} {r['name']}")
        print(f"  策略收益: {r['strategy_return']:+.2%}  (¥{r['strategy_value']:,.0f})")
        print(f"  持有收益: {r['buyhold_return']:+.2%}  (¥{r['buyhold_value']:,.0f})")
        print(f"  超额收益: {r['excess']:+.2%}")
        print(f"  交易次数: {r['trades']}")
        verdict = "优于" if r["excess"] > 0 else "劣于"
        print(f"  结论:     策略 {verdict} 持有不动")
        print()

    # Top 10 / Bottom 10
    sorted_results = sorted(results, key=lambda x: -x["excess"])

    print(f"{'─' * 72}")
    print(f"  超额收益 Top 10")
    print(f"{'─' * 72}")
    print(f"  {'代码':<8} {'名称':<24} {'策略收益':>9} {'持有收益':>9} {'超额':>9} {'交易':>4}")
    print(f"  {'─'*8} {'─'*24} {'─'*9} {'─'*9} {'─'*9} {'─'*4}")
    for r in sorted_results[:10]:
        star = " ★" if r["is_hs300"] else ""
        print(
            f"  {r['code']:<8} {r['name'][:24]:<24} "
            f"{r['strategy_return']:>+8.2%} {r['buyhold_return']:>+8.2%} "
            f"{r['excess']:>+8.2%} {r['trades']:>4}{star}"
        )

    print()
    print(f"{'─' * 72}")
    print(f"  超额收益 Bottom 10")
    print(f"{'─' * 72}")
    print(f"  {'代码':<8} {'名称':<24} {'策略收益':>9} {'持有收益':>9} {'超额':>9} {'交易':>4}")
    print(f"  {'─'*8} {'─'*24} {'─'*9} {'─'*9} {'─'*9} {'─'*4}")
    for r in sorted_results[-10:]:
        star = " ★" if r["is_hs300"] else ""
        print(
            f"  {r['code']:<8} {r['name'][:24]:<24} "
            f"{r['strategy_return']:>+8.2%} {r['buyhold_return']:>+8.2%} "
            f"{r['excess']:>+8.2%} {r['trades']:>4}{star}"
        )

    # 平均
    avg_strat = float(np.mean([r["strategy_return"] for r in results]))
    avg_buyhold = float(np.mean([r["buyhold_return"] for r in results]))
    avg_excess = float(np.mean([r["excess"] for r in results]))
    avg_trades = float(np.mean([r["trades"] for r in results]))

    print()
    print(f"{'─' * 72}")
    print(f"  汇总统计")
    print(f"{'─' * 72}")
    print(f"  平均策略收益: {avg_strat:+.2%}")
    print(f"  平均持有收益: {avg_buyhold:+.2%}")
    print(f"  平均超额收益: {avg_excess:+.2%}")
    print(f"  平均交易次数: {avg_trades:.1f}")
    print(f"  策略胜率:     {len(wins)}/{len(results)} = {len(wins)/len(results)*100:.1f}%")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
