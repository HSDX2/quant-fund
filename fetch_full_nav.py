#!/usr/bin/env python3
"""一次性脚本：为验证所需的 100 只基金拉取完整历史净值（不截断）。

nav_cache 使用 UPSERT 语义，增量数据会自动合并。
"""

import random
import sys
import os
import sqlite3
import json
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import akshare as ak
import nav_cache

DB_PATH = os.path.join(PROJECT_DIR, "cache", "nav_cache.db")


def load_universe():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT value_json FROM kv_cache WHERE key = 'filtered_universe'"
    ).fetchone()
    conn.close()
    return json.loads(row[0])


def fetch_full_nav(code):
    """拉取基金成立以来全部历史净值（不截断）。"""
    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df is None or df.empty:
            return None
        records = []
        for _, row in df.iterrows():
            date_str = row.iloc[0]
            if hasattr(date_str, "strftime"):
                date_str = date_str.strftime("%Y-%m-%d")
            else:
                date_str = str(date_str)[:10]
            records.append({"date": date_str, "nav": float(row.iloc[1])})
        records.sort(key=lambda x: x["date"])
        return records
    except Exception as e:
        print(f"  [ERROR] {code}: {e}")
        return None


def main():
    random.seed(42)

    universe = load_universe()
    print(f"基金池: {len(universe)} 只")

    # 和 validate_strategy.py 完全一致的选择逻辑
    hs300_code = "005918"
    other_funds = [f for f in universe if f["code"] != hs300_code]
    selected = random.sample(other_funds, min(99, len(other_funds)))
    hs300_fund = next((f for f in universe if f["code"] == hs300_code), None)
    if hs300_fund is None:
        hs300_fund = {"code": hs300_code, "name": "天弘沪深300ETF联接C", "type": "ETF联接"}
    selected.insert(0, hs300_fund)

    print(f"待拉取: {len(selected)} 只基金（完整历史净值）")
    print()

    success = 0
    fail = 0
    for idx, fund in enumerate(selected):
        code = fund["code"]
        name = fund.get("name", code)
        records = fetch_full_nav(code)
        if records and len(records) > 0:
            nav_cache.save_nav(code, records)
            success += 1
            min_d = records[0]["date"]
            max_d = records[-1]["date"]
            print(f"  [{idx+1}/{len(selected)}] {code} {name[:20]}: {len(records)} 条 ({min_d} ~ {max_d})")
        else:
            fail += 1
            print(f"  [{idx+1}/{len(selected)}] {code} {name[:20]}: 拉取失败")
        time.sleep(0.3)  # 避免 API 限流

    print(f"\n完成: 成功 {success}, 失败 {fail}")


if __name__ == "__main__":
    main()
