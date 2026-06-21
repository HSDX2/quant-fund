"""NAV 历史净值缓存 — SQLite 实现。

替代原有的 2604 个 JSON 文件，用单个 .db 文件存储所有基金净值数据。

优势：
  - 单文件 I/O，网络挂载盘上读写速度提升数十倍
  - 索引查询，按代码/日期范围快速检索
  - 批量读写，单事务提交全部记录
  - UPSERT 语义，重复拉取自动去重

Schema:
  nav_history(code, date, nav)  — 主键 (code, date)
  nav_meta(code, last_fetch, min_date, max_date, record_count)
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), "cache", "nav_cache.db")


def _get_conn() -> sqlite3.Connection:
    """获取数据库连接，自动建表。"""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, timeout=10)  # busy_timeout 10s
    conn.execute("PRAGMA journal_mode=WAL")  # 并发读不阻塞
    conn.execute("PRAGMA synchronous=NORMAL")  # 平衡安全与速度
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nav_history (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            nav REAL NOT NULL,
            PRIMARY KEY (code, date)
        );
        CREATE INDEX IF NOT EXISTS idx_nav_code ON nav_history(code);

        CREATE TABLE IF NOT EXISTS nav_meta (
            code TEXT PRIMARY KEY,
            last_fetch TEXT,
            min_date TEXT,
            max_date TEXT,
            record_count INTEGER
        );
    """)
    return conn


def get_nav(code: str, days: int = 1000) -> Optional[list[dict]]:
    """从缓存读取单只基金的净值数据。

    Returns:
        [{date, nav}, ...] 按日期升序，或 None（缓存不存在）
    """
    conn = _get_conn()
    try:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT date, nav FROM nav_history WHERE code = ? AND date >= ? ORDER BY date",
            (code, cutoff),
        ).fetchall()
        if not rows:
            return None
        return [{"date": r[0], "nav": r[1]} for r in rows]
    finally:
        conn.close()


def get_nav_batch(codes: list[str], days: int = 1000) -> dict[str, list[dict]]:
    """批量读取多只基金净值（单次查询）。

    Returns:
        {code: [{date, nav}, ...], ...}
    """
    if not codes:
        return {}

    conn = _get_conn()
    try:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        placeholders = ",".join("?" * len(codes))
        rows = conn.execute(
            f"SELECT code, date, nav FROM nav_history "
            f"WHERE code IN ({placeholders}) AND date >= ? ORDER BY code, date",
            (*codes, cutoff),
        ).fetchall()

        result: dict[str, list[dict]] = {}
        for code, date, nav in rows:
            if code not in result:
                result[code] = []
            result[code].append({"date": date, "nav": nav})
        return result
    finally:
        conn.close()


def save_nav(code: str, records: list[dict]) -> None:
    """保存单只基金净值到缓存（UPSERT）。"""
    if not records:
        return
    conn = _get_conn()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO nav_history (code, date, nav) VALUES (?, ?, ?)",
            [(code, r["date"], r["nav"]) for r in records],
        )
        dates = [r["date"] for r in records]
        conn.execute(
            "INSERT OR REPLACE INTO nav_meta (code, last_fetch, min_date, max_date, record_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (code, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             min(dates), max(dates), len(records)),
        )
        conn.commit()
    finally:
        conn.close()


def save_nav_batch(data: dict[str, list[dict]]) -> int:
    """批量保存多只基金净值到缓存（单事务，UPSERT）。

    Args:
        data: {code: [{date, nav}, ...], ...}
    Returns:
        保存的基金数量
    """
    if not data:
        return 0
    conn = _get_conn()
    count = 0
    try:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for code, records in data.items():
            if not records:
                continue
            conn.executemany(
                "INSERT OR REPLACE INTO nav_history (code, date, nav) VALUES (?, ?, ?)",
                [(code, r["date"], r["nav"]) for r in records],
            )
            dates = [r["date"] for r in records]
            conn.execute(
                "INSERT OR REPLACE INTO nav_meta (code, last_fetch, min_date, max_date, record_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (code, now_str, min(dates), max(dates), len(records)),
            )
            count += 1
        conn.commit()
    finally:
        conn.close()
    return count


def is_cache_valid(code: str, days: int = 1000) -> bool:
    """检查缓存是否有效。

    判据：
      1. min_date <= cutoff（覆盖所需天数）
      2. max_date >= today（已包含今天的数据——缓存命中）
         OR last_fetch 在 4 小时内（刚拉过，NAV 大概率未公布，避免重复请求）

    场景：
      - 晚上运行拉到今天数据 → max_date=today → 后续运行缓存命中
      - 早上运行未拉到今天数据 → last_fetch=now → 4h 内再运行不重复拉
      - 4h 后再运行（如晚上）→ 冷却期过 → 重新拉取，获取当天新数据
    """
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT last_fetch, min_date, max_date FROM nav_meta WHERE code = ?",
            (code,),
        ).fetchone()
        if not row:
            return False
        last_fetch, min_date, max_date = row
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")

        if min_date > cutoff:
            return False
        if max_date >= today:
            return True

        # max_date < today：可能还有新数据。检查是否在冷却期内。
        last_dt = _parse_dt(last_fetch)
        if last_dt and (datetime.now() - last_dt).total_seconds() < 4 * 3600:
            return True

        return False
    finally:
        conn.close()


def _parse_dt(s: str) -> datetime | None:
    """解析时间字符串，兼容 'YYYY-MM-DD HH:MM:SS' 和 'YYYY-MM-DD' 两种格式。"""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def batch_check_valid(codes: list[str], days: int = 1000) -> dict[str, bool]:
    """批量检查多只基金的缓存有效性（单次查询）。

    判据：min_date <= cutoff 且（max_date >= today 或 last_fetch 在 4h 冷却期内）。

    Returns:
        {code: bool}
    """
    if not codes:
        return {}
    conn = _get_conn()
    try:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now()
        placeholders = ",".join("?" * len(codes))
        rows = conn.execute(
            f"SELECT code, last_fetch, min_date, max_date FROM nav_meta WHERE code IN ({placeholders})",
            codes,
        ).fetchall()
        result = {}
        for code, last_fetch, min_date, max_date in rows:
            if min_date > cutoff:
                result[code] = False
            elif max_date >= today:
                result[code] = True
            else:
                last_dt = _parse_dt(last_fetch)
                result[code] = bool(last_dt and (now - last_dt).total_seconds() < 4 * 3600)
        for code in codes:
            if code not in result:
                result[code] = False
        return result
    finally:
        conn.close()


def get_all_cached_codes() -> set[str]:
    """获取所有有缓存的基金代码。"""
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT code FROM nav_meta").fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


def migrate_from_json(cache_dir: str) -> int:
    """从旧 JSON 缓存迁移到 SQLite（一次性迁移）。

    Args:
        cache_dir: cache 目录路径
    Returns:
        迁移的基金数量
    """
    import glob
    import json

    nav_files = glob.glob(os.path.join(cache_dir, "nav_*.json"))
    if not nav_files:
        logger.info("无 JSON 缓存需要迁移")
        return 0

    conn = _get_conn()
    count = 0
    try:
        for nav_path in nav_files:
            code = os.path.basename(nav_path).replace("nav_", "").replace(".json", "")
            try:
                with open(nav_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                if not records or not isinstance(records, list):
                    continue

                conn.executemany(
                    "INSERT OR REPLACE INTO nav_history (code, date, nav) VALUES (?, ?, ?)",
                    [(code, r["date"], r["nav"]) for r in records],
                )
                dates = [r["date"] for r in records]
                mtime = datetime.fromtimestamp(os.path.getmtime(nav_path))
                conn.execute(
                    "INSERT OR REPLACE INTO nav_meta (code, last_fetch, min_date, max_date, record_count) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (code, mtime.strftime("%Y-%m-%d"),
                     min(dates), max(dates), len(records)),
                )
                count += 1
                if count % 500 == 0:
                    conn.commit()
                    logger.info("  迁移进度: %d/%d", count, len(nav_files))
            except (json.JSONDecodeError, OSError, KeyError):
                continue

        conn.commit()
    finally:
        conn.close()

    logger.info("迁移完成: %d 只基金", count)
    return count
