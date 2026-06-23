#!/usr/bin/env python3
"""为老基金列表拉取完整历史净值（不截断）。"""
import json
import os
import sys
import time

PROJECT_DIR = r"D:\software\USTC-Drive\科大云盘\charming-cloud\project\quant-fund"
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import akshare as ak
import nav_cache

with open(os.path.join(PROJECT_DIR, "old_funds.json"), "r", encoding="utf-8") as f:
    funds = json.load(f)

print(f"待拉取: {len(funds)} 只基金（完整历史净值）")
print()

success = 0
fail = 0
for idx, fund in enumerate(funds):
    code = fund["code"]
    name = fund.get("name", code)
    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df is None or df.empty:
            print(f"  [{idx+1}/{len(funds)}] {code} {name[:20]}: 空数据")
            fail += 1
            continue
        records = []
        for _, row in df.iterrows():
            date_str = row.iloc[0]
            if hasattr(date_str, "strftime"):
                date_str = date_str.strftime("%Y-%m-%d")
            else:
                date_str = str(date_str)[:10]
            records.append({"date": date_str, "nav": float(row.iloc[1])})
        records.sort(key=lambda x: x["date"])
        nav_cache.save_nav(code, records)
        success += 1
        print(f"  [{idx+1}/{len(funds)}] {code} {name[:20]}: {len(records)} 条 ({records[0]['date']} ~ {records[-1]['date']})")
    except Exception as e:
        fail += 1
        print(f"  [{idx+1}/{len(funds)}] {code} {name[:20]}: 失败 {e}")
    time.sleep(0.5)

print(f"\n完成: 成功 {success}, 失败 {fail}")
