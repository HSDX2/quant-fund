import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import akshare as ak
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, config):
        self.config = config
        self.request_interval = 0.3

    def fetch_fund_list(self):
        import cache_db
        refresh_days = self.config["cache"]["universe_refresh_days"]

        if cache_db.is_kv_valid("fund_list", refresh_days):
            return cache_db.get_kv("fund_list")

        print("  正在获取全量基金列表（首次较慢）...")
        df = ak.fund_name_em()
        records = df.to_dict("records")
        cache_db.save_kv("fund_list", records)
        return records

    def fetch_fund_nav(self, code, days=300):
        """获取单只基金净值，使用 SQLite 缓存。"""
        import nav_cache

        # 缓存有效（max_date >= today 或冷却期内）→ 直接用缓存
        if nav_cache.is_cache_valid(code, days):
            return nav_cache.get_nav(code, days)

        # 需要从 API 拉取
        data = self._try_fetch_nav(code, days)
        if data is not None:
            nav_cache.save_nav(code, data)
            return data

        # API 失败时降级返回缓存（即使旧或覆盖不足）
        return nav_cache.get_nav(code, days)

    def _try_fetch_nav(self, code, days):
        """从 API 拉取单只基金净值（不写缓存，由调用方批量保存）。"""
        try:
            df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
            if df is None or df.empty:
                return None

            records = []
            for _, row in df.iterrows():
                date_str = row.iloc[0]
                if isinstance(date_str, datetime):
                    date_str = date_str.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_str)[:10]
                records.append({
                    "date": date_str,
                    "nav": float(row.iloc[1])
                })

            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            records = [r for r in records if r["date"] >= cutoff]
            records.sort(key=lambda x: x["date"])
            return records

        except Exception:
            return None

    def fetch_nav_batch(self, codes, days=300, max_workers=8):
        """批量获取多只基金净值。

        优化：先用 SQLite 批量读取缓存，再并行拉取未命中的基金，
        最后单事务批量保存。
        """
        import nav_cache

        # 第一步：批量读取所有缓存 + 批量检查有效性
        cached_data = nav_cache.get_nav_batch(codes, days)
        valid_map = nav_cache.batch_check_valid(codes, days)

        results = {}
        uncached = []

        for code in codes:
            data = cached_data.get(code)
            if data and len(data) >= 30 and valid_map.get(code, False):
                results[code] = data
            else:
                uncached.append(code)

        if uncached:
            logger.info("  缓存命中: %d 只，需拉取: %d 只", len(results), len(uncached))

        if not uncached:
            return results

        # 预热 V8 引擎：akshare 内部使用 py_mini_racer（V8），
        # 多线程同时首次调用会崩溃，需在主线程中先初始化一次。
        fetched_data = {}
        warmup_code = uncached[0]
        remaining = uncached[1:]
        try:
            warmup_data = self._try_fetch_nav(warmup_code, days)
            if warmup_data and len(warmup_data) >= 30:
                fetched_data[warmup_code] = warmup_data
                results[warmup_code] = warmup_data
        except Exception:
            pass

        if not remaining:
            # 只有一只需拉取，预热已完成
            if fetched_data:
                saved = nav_cache.save_nav_batch(fetched_data)
                logger.info("  批量保存缓存: %d 只", saved)
            return results

        # 第二步：并行拉取未命中的基金
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._try_fetch_nav, code, days): code
                       for code in remaining}
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="  获取净值", unit="只"
            ):
                code = futures[future]
                try:
                    data = future.result()
                    if data and len(data) >= 30:
                        fetched_data[code] = data
                        results[code] = data
                except Exception:
                    pass

        # 第三步：单事务批量保存到 SQLite
        if fetched_data:
            saved = nav_cache.save_nav_batch(fetched_data)
            logger.info("  批量保存缓存: %d 只", saved)

        return results

    @staticmethod
    def _fetch_single_estimate(code, max_retries=2):
        """获取单只基金的盘中实时估值（天天基金接口）。

        带重试机制，应对 fundgz 瞬时不可用。
        返回值:
            dict  — 成功获取估值数据
            None  — fundgz 明确无数据（如部分 QDII），无需重试
        """
        url = f"http://fundgz.1234567.com.cn/js/{code}.js"
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                resp = requests.get(url, timeout=5)
                resp.encoding = "utf-8"
                text = resp.text.strip()

                # fundgz 对不支持估值的基金返回空回调 jsonpgz();
                if text == "jsonpgz();" or len(text) <= 12:
                    return None  # 明确无数据，不重试

                json_str = text[text.index("{") : text.rindex("}") + 1]
                data = json.loads(json_str)
                return {
                    "code": data.get("fundcode", code),
                    "name": data.get("name", code),
                    "nav_yesterday": float(data.get("dwjz", 0)),
                    "estimate_nav": float(data.get("gsz", 0)),
                    "estimate_change": float(data.get("gszzl", 0)),
                    "estimate_time": data.get("gztime", ""),
                }
            except (ValueError, KeyError):
                # 解析失败 → 不重试
                logger.debug("fundgz 解析失败 %s: %.80s", code, text)
                return None
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue

        logger.warning("fundgz 获取失败 %s (重试%d次): %s", code, max_retries, last_error)
        return None

    def fetch_estimates(self, codes, max_workers=4):
        """并行获取多只基金的盘中实时估值。

        Returns:
            dict: {code: estimate_dict or None}, 只包含成功获取的条目
        """
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._fetch_single_estimate, code): code for code in codes}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="  获取估值", unit="只"
            ):
                code = futures[future]
                try:
                    est = future.result()
                    if est and est.get("estimate_nav", 0) > 0:
                        results[code] = est
                except Exception:
                    pass
        return results

    def fetch_fund_info_basic(self, code):
        """获取单只基金基本信息（成立时间 + 最新规模），永久缓存。

        使用雪球接口 ak.fund_individual_basic_info_xq()，
        一次调用返回 14 个字段含 成立时间、最新规模。

        Returns:
            dict or None: {"成立时间": "2018-04-24", "最新规模": 60.09, "基金代码": "005918", ...}
        """
        import cache_db
        cached = cache_db.get_fund_info(code)
        if cached:
            return cached

        time.sleep(self.request_interval)
        try:
            df = ak.fund_individual_basic_info_xq(symbol=code)
            if df is None or df.empty:
                return None

            # 雪球接口返回两列: item (字段名), value (字段值)
            record = {}
            for _, row in df.iterrows():
                key = str(row.iloc[0]).strip()
                val = str(row.iloc[1]).strip() if len(row) > 1 else ""
                record[key] = val

            # 解析规模: "60.09亿" → 60.09, 或 "5000万" → 0.5
            scale_str = record.get("最新规模", "")
            if scale_str:
                try:
                    if "亿" in scale_str:
                        record["_scale_yi"] = float(scale_str.replace("亿", ""))
                    elif "万" in scale_str:
                        record["_scale_yi"] = float(scale_str.replace("万", "")) / 10000
                except ValueError:
                    pass
            else:
                record["_scale_yi"] = None

            cache_db.save_fund_info(code, record)
            return record
        except Exception:
            return None

    def fetch_fund_info_batch(self, codes, max_workers=8):
        """批量获取多只基金基本信息（并行，永久缓存）。

        用于首次填充基金池的规模和成立时间数据。
        缓存命中直接返回，不发起网络请求。

        Returns:
            dict: {code: info_dict or None}
        """
        import cache_db
        results = cache_db.get_fund_info_batch(codes)
        uncached = [code for code in codes if results.get(code) is None]

        if not uncached:
            return results

        print(f"  获取基金基本信息: {len(uncached)} 只待拉取...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.fetch_fund_info_basic, code): code
                       for code in uncached}
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="  基本信息", unit="只"
            ):
                code = futures[future]
                try:
                    info = future.result()
                    results[code] = info
                except Exception:
                    results[code] = None
        return results

    def get_fund_name(self, code):
        fund_list = self.fetch_fund_list()
        for fund in fund_list:
            fund_code = str(fund.get("基金代码", "")).strip()
            if fund_code == code:
                return fund.get("基金简称", code)
        return code

    def fetch_fund_rank(self):
        print("  正在获取基金排名数据...")
        all_records = []
        for fund_type in ["股票型", "混合型", "债券型", "指数型", "QDII"]:
            try:
                time.sleep(0.5)
                df = ak.fund_open_fund_rank_em(symbol=fund_type)
                if df is not None and not df.empty:
                    records = df.to_dict("records")
                    for r in records:
                        r["_query_type"] = fund_type
                    all_records.extend(records)
                    print(f"    {fund_type}: {len(records)} 只")
            except Exception as e:
                print(f"    {fund_type}: 获取失败 ({e})")
        return all_records