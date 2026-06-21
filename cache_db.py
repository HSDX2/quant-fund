"""通用缓存 — SQLite 实现。

将项目中的其他缓存（基金列表、筛选池、基金基本信息）从 JSON 文件迁移到 SQLite，
减少网络挂载盘上的小文件 I/O。

基准指数数据复用 nav_cache 的 nav_history / nav_meta 表（结构相同：code, date, value）。

表结构:
  kv_cache(key, value_json, last_updated)  — 基金列表、筛选池等 KV 缓存
  fund_info(code, info_json)               — 基金基本信息（永久缓存）
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), "cache", "nav_cache.db")


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=10)  # busy_timeout 10s
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kv_cache (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL,
            last_updated TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fund_info (
            code TEXT PRIMARY KEY,
            info_json TEXT NOT NULL
        );
    """)
    return conn


# ── KV 缓存（基金列表、筛选池）──────────────────────────────

def get_kv(key: str):
    """读取 KV 缓存。返回 Python 对象或 None。"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT value_json FROM kv_cache WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])
    finally:
        conn.close()


def is_kv_valid(key: str, max_age_days: int) -> bool:
    """检查 KV 缓存是否存在且未过期。"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT last_updated FROM kv_cache WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return False
        last_updated = datetime.strptime(row[0][:19], "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - last_updated).days < max_age_days
    except (ValueError, TypeError):
        return False
    finally:
        conn.close()


def save_kv(key: str, value) -> None:
    """保存 KV 缓存。"""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO kv_cache (key, value_json, last_updated) VALUES (?, ?, ?)",
            (key, json.dumps(value, ensure_ascii=False),
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    finally:
        conn.close()


# ── 基金基本信息（永久缓存）──────────────────────────────────

def get_fund_info(code: str):
    """读取基金基本信息。返回 dict 或 None。"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT info_json FROM fund_info WHERE code = ?", (code,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])
    finally:
        conn.close()


def get_fund_info_batch(codes: list[str]) -> dict:
    """批量读取基金基本信息。返回 {code: dict or None}。"""
    if not codes:
        return {}
    conn = _get_conn()
    try:
        placeholders = ",".join("?" * len(codes))
        rows = conn.execute(
            f"SELECT code, info_json FROM fund_info WHERE code IN ({placeholders})",
            codes,
        ).fetchall()
        result = {code: None for code in codes}
        for code, info_json in rows:
            result[code] = json.loads(info_json)
        return result
    finally:
        conn.close()


def save_fund_info(code: str, info: dict) -> None:
    """保存基金基本信息（永久缓存，不设过期）。"""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO fund_info (code, info_json) VALUES (?, ?)",
            (code, json.dumps(info, ensure_ascii=False)),
        )
        conn.commit()
    finally:
        conn.close()


# ── 迁移 ──────────────────────────────────────────────────────

def migrate_json_kv(cache_dir: str, key_map: dict[str, str]) -> int:
    """将 JSON 文件迁移到 KV 缓存。

    Args:
        cache_dir: cache 目录路径
        key_map: {json_filename: kv_key}，如 {"fund_list.json": "fund_list"}
    Returns:
        迁移的条目数
    """
    conn = _get_conn()
    count = 0
    try:
        for filename, key in key_map.items():
            filepath = os.path.join(cache_dir, filename)
            if not os.path.exists(filepath):
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                conn.execute(
                    "INSERT OR REPLACE INTO kv_cache (key, value_json, last_updated) VALUES (?, ?, ?)",
                    (key, json.dumps(data, ensure_ascii=False),
                     mtime.strftime("%Y-%m-%d %H:%M:%S")),
                )
                count += 1
                logger.info("  迁移 KV: %s → %s", filename, key)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("  迁移失败: %s (%s)", filename, e)
        conn.commit()
    finally:
        conn.close()
    return count


def migrate_fund_info_json(cache_dir: str) -> int:
    """将 info_*.json 文件迁移到 fund_info 表。

    Args:
        cache_dir: cache 目录路径
    Returns:
        迁移的条目数
    """
    import glob
    info_files = glob.glob(os.path.join(cache_dir, "info_*.json"))
    if not info_files:
        logger.info("无 info_*.json 需要迁移")
        return 0

    conn = _get_conn()
    count = 0
    try:
        for filepath in info_files:
            code = os.path.basename(filepath).replace("info_", "").replace(".json", "")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    record = json.load(f)
                if not record or not isinstance(record, dict):
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO fund_info (code, info_json) VALUES (?, ?)",
                    (code, json.dumps(record, ensure_ascii=False)),
                )
                count += 1
                if count % 1000 == 0:
                    conn.commit()
                    logger.info("  迁移进度: %d/%d", count, len(info_files))
            except (json.JSONDecodeError, OSError, KeyError):
                continue
        conn.commit()
    finally:
        conn.close()
    logger.info("迁移完成: %d 只基金信息", count)
    return count
