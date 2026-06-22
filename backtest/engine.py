"""回测引擎：逐日推进，模拟忠实遵循基金报告的操作建议。

核心流程:
  对于每一天 t:
    1. 截断 nav_data 只保留 t 及之前的数据
    2. 清理尘埃持仓（价值 < 总资产 0.1%）
    3. 计算技术指标
    4. 生成买卖信号（与实盘 main.py 完全一致的代码路径）
    5. 执行持仓操作建议（清仓/减仓）
    6. 买入排名第一的未持仓基金（持仓 < 30 时）
    7. 记录当日快照

与实盘的一致性:
  - 使用 analyze_holdings() + compute_position_advice() 生成建议，
    与 main.py 的 run_pipeline() 完全一致
  - 每日最多买入 1 只基金（排名第一的未持仓基金）
  - 卖出比例严格遵循 hold_positions 中的 pct
  - 尘埃持仓清理阈值：总资产的 0.1%
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional

from .portfolio import Portfolio, Trade, DailySnapshot, MAX_HOLDINGS

logger = logging.getLogger(__name__)


class BacktestEngine:
    """基金策略回测引擎。

    Args:
        config:         完整配置字典
        holding_codes:  初始持仓基金代码列表（仅用于策略的 holding 识别，
                        回测从纯现金开始）
        nav_data:       净值数据 {code: [{date, nav}, ...]}
        universe:       基金池 [{code, name, type, ...}, ...]
        start_date:     回测起点（None = 自动：最早可用数据 + 200天）
        end_date:       回测终点（None = 昨天）
        initial_capital:初始资金
    """

    # 尘埃持仓阈值：占总资产的比例
    DUST_THRESHOLD_PCT = 0.001  # 0.1%

    def __init__(
        self,
        config: dict,
        holding_codes: list[str],
        nav_data: dict,
        universe: list[dict],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_capital: float = 100000.0,
    ):
        self.config = config
        self.strategy = config["strategy"]
        self.holding_codes = holding_codes
        self.nav_data = nav_data
        self.universe = universe
        self.initial_capital = initial_capital

        # 构建辅助索引
        self._code_name = {f["code"]: f["name"] for f in universe}
        self._code_type = {f["code"]: f.get("type", "混合型") for f in universe}

        # 日期范围
        self.start_date = start_date
        self.end_date = end_date or datetime.now()
        self._trading_dates: list[datetime] = []
        self._determine_date_range()

        # 结果
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.rule_stats: dict = defaultdict(lambda: {"triggers": 0, "wins": 0})
        self.errors: list[str] = []

    # ── 日期范围 ──────────────────────────────────────

    def _determine_date_range(self):
        """确定回测日期范围，确保有足够的历史数据做 MA200。"""
        all_dates: set[datetime] = set()
        for records in self.nav_data.values():
            for r in records:
                d = r["date"]
                if isinstance(d, str):
                    d = datetime.strptime(d[:10], "%Y-%m-%d")
                all_dates.add(d)

        if not all_dates:
            raise ValueError("nav_data 为空，无法确定日期范围")

        sorted_dates = sorted(all_dates)

        earliest_possible = sorted_dates[200] if len(sorted_dates) > 200 else sorted_dates[0]
        if self.start_date is None:
            self.start_date = earliest_possible
        else:
            self.start_date = max(self.start_date, earliest_possible)

        self.end_date = min(self.end_date, sorted_dates[-1])

        self._trading_dates = [
            d for d in sorted_dates
            if self.start_date <= d <= self.end_date
        ]

        if len(self._trading_dates) < 30:
            raise ValueError(
                f"交易日不足（{len(self._trading_dates)}天），无法回测"
            )

        logger.info(
            "回测区间: %s ~ %s (%d 个交易日)",
            self.start_date.strftime("%Y-%m-%d"),
            self.end_date.strftime("%Y-%m-%d"),
            len(self._trading_dates),
        )

    # ── 数据截断（核心：无未来信息）──────────────────

    def _truncate_nav(self, cutoff_date: datetime) -> dict:
        """截断净值数据：只保留 cutoff_date 及之前的记录。"""
        truncated = {}
        for code, records in self.nav_data.items():
            filtered = []
            for r in records:
                d = r["date"]
                if isinstance(d, str):
                    d = datetime.strptime(d[:10], "%Y-%m-%d")
                if d <= cutoff_date:
                    filtered.append(r)
            if filtered:
                truncated[code] = filtered
        return truncated

    def _build_nav_dict(self, truncated_nav: dict) -> dict[str, float]:
        """从截断数据构建 {code: 最新净值} 字典。"""
        nav_dict = {}
        for code, records in truncated_nav.items():
            if records:
                nav_dict[code] = records[-1]["nav"]
        return nav_dict

    # ── 主循环 ────────────────────────────────────────

    def run(self) -> dict:
        """执行回测，返回汇总结果。"""
        from indicators import compute_indicators
        from scorer import select_buy_candidates, select_sell_candidates
        from holdings import analyze_holdings
        from position_advisor import compute_position_advice

        total = len(self._trading_dates)
        target_vol = self.strategy.get("target_annual_vol", 0.10)
        rules_cfg = self.strategy.get("rules")

        equity_codes = {
            code for code, ftype in self._code_type.items()
            if any(t in ftype for t in ("股票型", "混合型", "指数型", "QDII"))
        }

        for idx, today in enumerate(self._trading_dates):
            if idx % 50 == 0:
                logger.info(
                    "  回测进度: %d/%d (%.0f%%)",
                    idx, total, idx / total * 100,
                )

            # 1. 截断数据
            truncated_nav = self._truncate_nav(today)
            nav_dict = self._build_nav_dict(truncated_nav)

            # 2. 清理尘埃持仓（< 总资产 0.1%）
            self._cleanup_dust_positions(nav_dict, today)

            # 3. 当前持仓代码
            my_codes = list(self.portfolio.positions.keys())

            # 4. 计算指标
            indicators = compute_indicators(
                truncated_nav, self.strategy, equity_codes,
                set(my_codes),
            )

            # 5. 生成信号（与 main.py run_pipeline 完全一致）
            buy_top10 = select_buy_candidates(
                indicators, rules=rules_cfg, top_n=10,
            )
            sell_top10 = select_sell_candidates(
                indicators, holding_codes=my_codes, rules=rules_cfg,
            )
            holdings_advice = analyze_holdings(
                my_codes, indicators, self.strategy, rules=rules_cfg,
            )

            # 填充基金名称（与 main.py 一致）
            for item in buy_top10:
                item["name"] = self._code_name.get(item["code"], item["code"])
            for item in sell_top10:
                item["name"] = self._code_name.get(item["code"], item["code"])
            for item in holdings_advice:
                if item.get("name", "") == item["code"]:
                    item["name"] = self._code_name.get(item["code"], item["code"])

            buy_positions, hold_positions = compute_position_advice(
                indicators, truncated_nav, buy_top10, sell_top10,
                holdings_advice, target_vol=target_vol,
            )

            # 6. 执行卖出建议（清仓/减仓）
            self._execute_sell_advice(hold_positions, truncated_nav, today)

            # 7. 执行买入建议（排名第一的未持仓基金）
            self._execute_buy_advice(buy_positions, truncated_nav, today)

            # 8. 快照
            self.portfolio.snapshot(today, nav_dict)

        logger.info("  回测完成")
        return self._build_result()

    # ── 尘埃持仓清理 ──────────────────────────────────

    def _cleanup_dust_positions(
        self, nav_dict: dict[str, float], today: datetime,
    ) -> None:
        """清理价值低于总资产 0.1% 的持仓。"""
        if not self.portfolio.positions:
            return

        total_value = self.portfolio.total_value(nav_dict)
        threshold = total_value * self.DUST_THRESHOLD_PCT

        for code in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions[code]
            nav = nav_dict.get(code, 0)
            if nav <= 0:
                continue

            value = pos["units"] * nav
            if value < threshold:
                name = self._code_name.get(code, code)
                ftype = self._code_type.get(code, "混合型")
                self.portfolio.sell(
                    date=today, code=code, name=name,
                    nav=nav, units=1.0, fund_type=ftype,
                    reason=f"Dust cleanup (<{self.DUST_THRESHOLD_PCT:.0%} of total)",
                )

    # ── 卖出执行 ──────────────────────────────────────

    def _execute_sell_advice(
        self, hold_positions: list[dict],
        nav_data: dict, today: datetime,
    ) -> None:
        """执行持仓操作建议中的卖出动作。

        清仓 → 卖出 100%
        减仓 → 卖出 pct 比例的持仓
        持有 → 不操作
        """
        for item in hold_positions:
            code = item["code"]
            if code not in self.portfolio.positions:
                continue

            action = item.get("action", "")
            if action not in ("清仓", "减仓"):
                continue

            # 确定卖出比例
            if action == "清仓":
                sell_units = 1.0
            else:  # 减仓
                sell_units = item.get("pct", 0)
                if sell_units <= 0:
                    continue  # 无法计算减仓比例，跳过

            nav = item.get("nav", 0)
            if nav <= 0:
                records = nav_data.get(code, [])
                if records:
                    nav = records[-1]["nav"]
                if nav <= 0:
                    continue

            name = self._code_name.get(code, code)
            ftype = self._code_type.get(code, "混合型")
            reason = item.get("reason_detail") or item.get("reason", "")

            trade = self.portfolio.sell(
                date=today, code=code, name=name,
                nav=nav, units=sell_units, fund_type=ftype,
                reason=reason,
            )

            if trade:
                for rule_token in reason.split(" | "):
                    rule_id = rule_token.split(" ")[0] if rule_token else ""
                    if rule_id:
                        self.rule_stats[rule_id]["triggers"] += 1

    # ── 买入执行 ──────────────────────────────────────

    def _execute_buy_advice(
        self, buy_positions: list[dict],
        nav_data: dict, today: datetime,
    ) -> None:
        """执行买入：持仓 < 30 时，买入排名第一的未持仓基金。

        买入金额 = 总资产 × pct（pct 来自 compute_position_advice，
        与基金分析报告中的建议仓位比例完全一致）。
        """
        if self.portfolio.position_count() >= MAX_HOLDINGS:
            return

        # 过滤已持仓的，取排名第一
        unheld = [
            p for p in buy_positions
            if p["code"] not in self.portfolio.positions
        ]
        if not unheld:
            return

        candidate = unheld[0]  # buy_positions 已按 rebound 降序排列
        code = candidate["code"]
        pct = candidate.get("pct", 0)

        if pct <= 0:
            return

        nav = candidate.get("nav", 0)
        if nav <= 0:
            return

        # 买入金额 = 总资产 × pct
        nav_dict = self._build_nav_dict(nav_data)
        total_value = self.portfolio.total_value(nav_dict)
        amount = total_value * pct

        # 不超过可用现金的 95%
        available = self.portfolio.cash * 0.95
        amount = min(amount, available)

        if amount < 500:
            return

        name = self._code_name.get(code, code)
        ftype = self._code_type.get(code, "混合型")
        reason = candidate.get("reason", "")

        trade = self.portfolio.buy(
            date=today, code=code, name=name,
            nav=nav, amount=amount, fund_type=ftype,
            reason=reason,
        )

        if trade:
            for rule_token in reason.split(" | "):
                rule_id = rule_token.split(" ")[0] if rule_token else ""
                if rule_id:
                    self.rule_stats[rule_id]["triggers"] += 1

    # ── 结果汇总 ──────────────────────────────────────

    def _build_result(self) -> dict:
        """构建回测结果字典。"""
        stats = self.portfolio.stats()

        # 计算规则胜率
        LOOKAHEAD = 20  # 前瞻窗口（交易日）

        rule_buy_trades: dict[str, list[Trade]] = defaultdict(list)
        rule_sell_trades: dict[str, list[Trade]] = defaultdict(list)

        for trade in self.portfolio.trade_log:
            for rule_token in trade.reason.split(" | "):
                rule_id = rule_token.split(" ")[0] if rule_token else ""
                if not rule_id:
                    continue
                if trade.action == "买入":
                    rule_buy_trades[rule_id].append(trade)
                elif trade.action == "卖出":
                    rule_sell_trades[rule_id].append(trade)

        for rule_id, trades in rule_sell_trades.items():
            self.rule_stats[rule_id]["triggers"] = len(trades)
            for trade in trades:
                if self._post_sell_declined(trade, LOOKAHEAD):
                    self.rule_stats[rule_id]["wins"] += 1

        for rule_id, trades in rule_buy_trades.items():
            for trade in trades:
                if self._post_buy_rose(trade, LOOKAHEAD):
                    self.rule_stats[rule_id]["wins"] += 1

        rule_effectiveness = {}
        for rule_id, data in self.rule_stats.items():
            total_t = data["triggers"]
            wins = data["wins"]
            rule_effectiveness[rule_id] = {
                "triggers": total_t,
                "wins": wins,
                "win_rate": wins / total_t if total_t > 0 else 0.0,
            }

        # 交易过的基金表现
        traded_codes: set[str] = set()
        for trade in self.portfolio.trade_log:
            traded_codes.add(trade.code)

        fund_returns = {}
        for code in traded_codes:
            records = self.nav_data.get(code, [])
            if len(records) >= 2:
                start_nav = records[0]["nav"]
                end_nav = records[-1]["nav"]
                if start_nav > 0:
                    fund_returns[code] = end_nav / start_nav - 1

        top_funds = sorted(
            fund_returns.items(), key=lambda x: x[1], reverse=True
        )[:5]
        bottom_funds = sorted(
            fund_returns.items(), key=lambda x: x[1]
        )[:5]

        return {
            "stats": stats,
            "rule_effectiveness": rule_effectiveness,
            "top_performers": [
                {"code": c, "name": self._code_name.get(c, c), "return": r}
                for c, r in top_funds
            ],
            "worst_performers": [
                {"code": c, "name": self._code_name.get(c, c), "return": r}
                for c, r in bottom_funds
            ],
            "trade_log": self.portfolio.trade_log,
            "snapshots": self.portfolio.snapshots,
            "errors": self.errors,
            "initial_capital": self.initial_capital,
            "benchmark_stats": self._compute_benchmark(),
            "benchmark_name": "沪深300",
        }

    def _compute_benchmark(self) -> dict:
        """获取沪深300基准数据并计算对比指标。"""
        try:
            from benchmark import fetch_benchmark_data, compute_benchmark_stats
            benchmark_data = fetch_benchmark_data("000300", days=1000)
            return compute_benchmark_stats(
                benchmark_data, self.start_date, self.end_date
            )
        except Exception as e:
            logger.warning("基准对比计算失败: %s", e)
            return {}

    def _post_sell_declined(self, trade: Trade, lookahead: int) -> bool:
        """卖出后 lookahead 日内基金是否下跌（卖出正确）。"""
        records = self.nav_data.get(trade.code, [])
        sell_nav = trade.nav
        future_navs = []
        for r in records:
            d = r["date"]
            if isinstance(d, str):
                d = datetime.strptime(d[:10], "%Y-%m-%d")
            if d > trade.date:
                future_navs.append(r["nav"])

        if len(future_navs) < lookahead:
            if future_navs:
                return future_navs[-1] < sell_nav
            return False

        future_nav = future_navs[min(lookahead - 1, len(future_navs) - 1)]
        return future_nav < sell_nav

    def _post_buy_rose(self, trade: Trade, lookahead: int) -> bool:
        """买入后 lookahead 日内基金是否上涨（买入正确）。"""
        records = self.nav_data.get(trade.code, [])
        buy_nav = trade.nav
        future_navs = []
        for r in records:
            d = r["date"]
            if isinstance(d, str):
                d = datetime.strptime(d[:10], "%Y-%m-%d")
            if d > trade.date:
                future_navs.append(r["nav"])

        if len(future_navs) < lookahead:
            if future_navs:
                return future_navs[-1] > buy_nav
            return False

        future_nav = future_navs[min(lookahead - 1, len(future_navs) - 1)]
        return future_nav > buy_nav
