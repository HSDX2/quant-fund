#!/usr/bin/env python3
"""选取200只基金并拉取完整历史净值。"""
import random
import json
import os
import sys
import time

PROJECT_DIR = r"D:\software\USTC-Drive\科大云盘\charming-cloud\project\quant-fund"
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import akshare as ak
import nav_cache
import sqlite3

DB = os.path.join(PROJECT_DIR, "cache", "nav_cache.db")

# 加载基金池
conn = sqlite3.connect(DB)
row = conn.execute("SELECT value_json FROM kv_cache WHERE key = 'filtered_universe'").fetchone()
universe = json.loads(row[0])

# 查已有完整历史的基金
rows = conn.execute("""
    SELECT code, MIN(date) FROM nav_history GROUP BY code
    HAVING MIN(date) <= '2021-06-01'
""").fetchall()
existing = {r[0] for r in rows}
conn.close()

# 分两组：已有完整历史 + 需要拉取
need_fetch = [f for f in universe if f["code"] not in existing]
already = [f for f in universe if f["code"] in existing]

print(f"基金池: {len(universe)} 只")
print(f"已有完整历史: {len(already)} 只")
print(f"需要拉取: {len(need_fetch)} 只")

# 随机选200只
random.seed(2024)
# 优先选已有的
selected = random.sample(already, min(len(already), 80))
# 补充需要拉取的
remaining = 200 - len(selected)
if remaining > 0:
    extra = random.sample(need_fetch, min(remaining, len(need_fetch)))
    selected.extend(extra)

print(f"选中: {len(selected)} 只 (已有:{len(already) and min(len(already), 80)}, 需拉取:{len(selected) - min(len(already), 80)})")

# 保存选中基金列表
with open(os.path.join(PROJECT_DIR, "sample200.json"), "w", encoding="utf-8") as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)

# 拉取需要补数据的基金
to_fetch = [f for f in selected if f["code"] not in existing]
print(f"\n开始拉取 {len(to_fetch)} 只基金的完整历史净值...")

success = 0
fail = 0
for idx, fund in enumerate(to_fetch):
    code = fund["code"]
    name = fund.get("name", code)
    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df is None or df.empty:
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
        if (idx + 1) % 20 == 0:
            print(f"  进度: {idx+1}/{len(to_fetch)} (成功:{success}, 失败:{fail})")
    except Exception as e:
        fail += 1
    time.sleep(0.3)

print(f"\n拉取完成: 成功 {success}, 失败 {fail}")
print(f"共 {len(selected)} 只基金可用于回测")
