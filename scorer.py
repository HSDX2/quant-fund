# fund-metrics rule library
#
# Buy rules (match any to enter candidate pool, ranked by composite score):
#   P1 Golden Pullback:   above MA200, decline 2-7d, RSI < 45
#   P2 Oversold Bounce:   above MA200, RSI < 35, 60d DD > 6%
#   P3 Trend Pullback:    above MA200, decline 2-7d, Sharpe > 0.3
#   P4 Low-Vol Dip:       above MA200, RSI < 40, vol < historical
#   P5 Deep Value:        above MA200, 60d DD > 10%, RSI < 40
#   P6 Quality Dip:       above MA200, Sharpe > 0.5, decline 1-7d
#   P7 Vol Contraction:   above MA200, vol < 0.6x hist, pullback > 2%
#   P8 Strong Trend Dip:  above MA200, trend > 3%, decline >= 1d
#
# Sell rules (all checked, collect all triggers):
#   S1 Hard Stop:         60d max drawdown > 15%              → 清仓
#   S2 Consecutive Rise:  rise >= 5d, RSI > 75                → 减仓
#   S3 Surge Profit:      monthly > 12%, RSI > 80             → 减仓
#   S4 Risk Decay:        Sharpe < -0.5, monthly < -8%        → 减仓
#   S5 Peak Retreat:      above MA200, pullback > 8%, decl >=5d → 减仓
#   S6 Deviation:         trend > 15%, RSI > 70               → 减仓
#   S7 Vol Spike:         vol > 2.0x historical               → 减仓
#   S8 Trend Confirm:     below MA200, monthly < -5%          → 清仓
#
# 所有规则阈值可从 config.yaml 的 strategy.rules 段覆盖。
# 传入 rules=None 时使用下方的默认值（向后兼容）。


# ── 默认规则参数 ──────────────────────────────────────

_DEFAULT_BUY_RULES = {
    "P1_golden_pullback":    {"enabled": True, "rsi_max": 45, "decline_min": 2, "decline_max": 7},
    "P2_oversold_bounce":    {"enabled": True, "rsi_max": 35, "max_drawdown_min": 0.06},
    "P3_trend_pullback":     {"enabled": True, "decline_min": 2, "decline_max": 7, "sharpe_min": 0.3},
    "P4_low_vol_dip":        {"enabled": True, "rsi_max": 40, "vol_ratio_max": 1.0},
    "P5_deep_value":         {"enabled": True, "max_drawdown_min": 0.10, "rsi_max": 40},
    "P6_quality_dip":        {"enabled": True, "sharpe_min": 0.5, "decline_min": 1, "decline_max": 7},
    "P7_vol_contraction":    {"enabled": True, "vol_ratio_max": 0.6, "pullback_min": 0.02},
    "P8_strong_trend_dip":   {"enabled": True, "trend_strength_min": 0.03, "decline_min": 1},
}

_DEFAULT_SELL_RULES = {
    "S1_hard_stop":          {"enabled": True, "max_drawdown_min": 0.15},
    "S2_consecutive_rise":   {"enabled": True, "rise_min": 5, "rsi_min": 75},
    "S3_surge_profit":       {"enabled": True, "monthly_return_min": 0.12, "rsi_min": 80},
    "S4_risk_decay":         {"enabled": True, "sharpe_max": -0.5, "monthly_return_max": -0.08},
    "S5_peak_retreat":       {"enabled": True, "pullback_min": 0.08, "decline_min": 5},
    "S6_deviation":          {"enabled": True, "trend_strength_min": 0.15, "rsi_min": 70},
    "S7_vol_spike":          {"enabled": True, "vol_ratio_min": 2.0},
    "S8_trend_confirm":      {"enabled": True, "monthly_return_max": -0.05},
}


def _get_rules(user_rules, section, rule_id):
    """获取某条规则的参数，优先用用户配置，回退到默认值。"""
    if user_rules:
        rule = user_rules.get(section, {}).get(rule_id, {})
        if rule:
            return rule
    defaults = (
        _DEFAULT_BUY_RULES if section == "buy" else _DEFAULT_SELL_RULES
    )
    return defaults.get(rule_id, {})


# ── 买入候选 ─────────────────────────────────────────

def select_buy_candidates(indicators, rules=None, top_n=30):
    """Rule-based buy candidate selection, ranked by multi-factor composite score.

    排序因子：
      - 跌幅得分 (30%):  回调幅度越大，反弹空间越大
      - 趋势得分 (30%):  趋势强度越高，基本面越好
      - 夏普得分 (20%):  风险调整收益越高，基金质量越好
      - RSI 得分 (20%):  RSI 越低，超卖程度越深
    """
    candidates = []

    for code, sig in indicators.items():
        reason = None

        # P1: Golden Pullback
        r = _get_rules(rules, "buy", "P1_golden_pullback")
        if r.get("enabled", True) and (sig["above_ma200"]
                and r["decline_min"] <= sig["consecutive_declines"] <= r["decline_max"]
                and sig["rsi_14"] < r["rsi_max"]):
            reason = f"P1 Golden pullback ({sig['consecutive_declines']}d, RSI={sig['rsi_14']:.0f})"

        # P2: Oversold Bounce
        elif not reason:
            r = _get_rules(rules, "buy", "P2_oversold_bounce")
            if r.get("enabled", True) and (sig["above_ma200"]
                    and sig["rsi_14"] < r["rsi_max"]
                    and sig["rolling_max_drawdown"] > r["max_drawdown_min"]):
                reason = f"P2 Oversold bounce (RSI={sig['rsi_14']:.0f}, DD={sig['rolling_max_drawdown']:.1%})"

        # P3: Trend Pullback
        elif not reason:
            r = _get_rules(rules, "buy", "P3_trend_pullback")
            if r.get("enabled", True) and (sig["above_ma200"]
                    and r["decline_min"] <= sig["consecutive_declines"] <= r["decline_max"]
                    and sig["rolling_sharpe"] > r["sharpe_min"]):
                reason = f"P3 Trend pullback ({sig['consecutive_declines']}d, Sharpe={sig['rolling_sharpe']:.1f})"

        # P4: Low-Vol Dip
        elif not reason:
            r = _get_rules(rules, "buy", "P4_low_vol_dip")
            if r.get("enabled", True) and (sig["above_ma200"]
                    and sig["rsi_14"] < r["rsi_max"]
                    and sig["volatility_ratio"] < r["vol_ratio_max"]):
                reason = f"P4 Low-vol dip (RSI={sig['rsi_14']:.0f}, vol<hist)"

        # P5: Deep Value
        elif not reason:
            r = _get_rules(rules, "buy", "P5_deep_value")
            if r.get("enabled", True) and (sig["above_ma200"]
                    and sig["rolling_max_drawdown"] > r["max_drawdown_min"]
                    and sig["rsi_14"] < r["rsi_max"]):
                reason = f"P5 Deep value (DD={sig['rolling_max_drawdown']:.1%}, RSI={sig['rsi_14']:.0f})"

        # P6: Quality Dip
        elif not reason:
            r = _get_rules(rules, "buy", "P6_quality_dip")
            if r.get("enabled", True) and (sig["above_ma200"]
                    and sig["rolling_sharpe"] > r["sharpe_min"]
                    and r["decline_min"] <= sig["consecutive_declines"] <= r["decline_max"]):
                reason = f"P6 Quality dip ({sig['consecutive_declines']}d, Sharpe={sig['rolling_sharpe']:.1f})"

        # P7: Vol Contraction
        elif not reason:
            r = _get_rules(rules, "buy", "P7_vol_contraction")
            if r.get("enabled", True) and (sig["above_ma200"]
                    and sig["volatility_ratio"] < r["vol_ratio_max"]
                    and sig["pullback_from_peak"] > r["pullback_min"]):
                reason = f"P7 Vol contraction (vol {sig['volatility_ratio']:.2f}x, pullback {sig['pullback_from_peak']:.1%})"

        # P8: Strong Trend Dip
        elif not reason:
            r = _get_rules(rules, "buy", "P8_strong_trend_dip")
            if r.get("enabled", True) and (sig["above_ma200"]
                    and sig["trend_strength"] > r["trend_strength_min"]
                    and sig["consecutive_declines"] >= r["decline_min"]):
                reason = f"P8 Strong trend dip ({sig['consecutive_declines']}d, trend +{sig['trend_strength']:.1%})"

        if reason:
            # 多因子综合评分
            vol = max(sig["volatility_ratio"], 0.01)

            decline_score = min(abs(sig["recent_decline"]) / 0.10, 1.0)
            trend_score = min(max(sig["trend_strength"], 0) / 0.10, 1.0)
            sharpe_score = max(0, min(sig["rolling_sharpe"] / 2.0, 1.0))
            rsi_score = max(0, (50 - sig["rsi_14"]) / 50)

            composite = (
                decline_score * 0.3
                + trend_score * 0.3
                + sharpe_score * 0.2
                + rsi_score * 0.2
            )

            candidates.append({
                "code": code,
                "reason": reason,
                "nav": sig["current_nav"],
                "trend_strength": sig["trend_strength"],
                "rebound": composite,  # 保留字段名兼容
            })

    candidates.sort(key=lambda x: x["rebound"], reverse=True)
    return candidates[:top_n]


# ── 卖出候选 ─────────────────────────────────────────

def select_sell_candidates(indicators, holding_codes=None, rules=None):
    """Rule-based sell candidate selection, ranked by severity.

    统一调用 classify_holding()，消除重复逻辑。
    """
    candidates = []
    codes = holding_codes if holding_codes else list(indicators.keys())

    for code in codes:
        if code not in indicators:
            continue
        sig = indicators[code]

        action, reason = classify_holding(sig, rules=rules)
        if action in ("清仓", "减仓"):
            severity = _sell_severity(sig)
            candidates.append({
                "code": code,
                "action": action,
                "reason": reason,
                "nav": sig["current_nav"],
                "severity": severity,
            })

    candidates.sort(key=lambda x: x["severity"], reverse=True)
    return candidates[:10]


def _sell_severity(sig):
    """Sell severity (0-1): max of four normalized risk dimensions."""
    trend = max(0.0, -sig["trend_strength"]) / 0.05 if not sig["above_ma200"] else 0.0
    drawdown = sig["rolling_max_drawdown"] / 0.15
    vol = max(0.0, sig["volatility_ratio"] - 1.0) / 1.0
    rsi = max(0.0, sig["rsi_14"] - 60) / 40

    return round(min(1.0, max(trend, drawdown, vol, rsi)), 4)


# ── 持仓分类（唯一规则判定入口）─────────────────────

def classify_holding(sig, rules=None):
    """Classify a single holding using rules.  Returns (action, reason).

    检查全部卖出规则，收集所有触发的规则。
    若任一清仓规则触发 → action = "清仓"。
    否则若有减仓规则触发 → action = "减仓"。
    否则 → "持有"。

    不再包含加仓逻辑（回测中加仓由买入候选列表处理）。
    """
    triggers = []
    has_clear = False

    # S1: 硬止损 — 严重回撤，论点失效
    r = _get_rules(rules, "sell", "S1_hard_stop")
    if r.get("enabled", True) and sig["rolling_max_drawdown"] > r["max_drawdown_min"]:
        triggers.append(f"S1 Hard stop (DD {sig['rolling_max_drawdown']:.1%})")
        has_clear = True

    # S2: 连涨止盈 — 连续上涨后过热
    r = _get_rules(rules, "sell", "S2_consecutive_rise")
    if r.get("enabled", True) and sig["consecutive_rises"] >= r["rise_min"] and sig["rsi_14"] > r["rsi_min"]:
        triggers.append(
            f"S2 Overbought ({sig['consecutive_rises']}d rise, RSI={sig['rsi_14']:.0f})"
        )

    # S3: 暴涨止盈 — 月涨幅过大，锁定利润
    r = _get_rules(rules, "sell", "S3_surge_profit")
    if r.get("enabled", True) and sig["monthly_return"] > r["monthly_return_min"] and sig["rsi_14"] > r["rsi_min"]:
        triggers.append(
            f"S3 Surge profit (monthly {sig['monthly_return']:.1%}, RSI={sig['rsi_14']:.0f})"
        )

    # S4: 风险恶化 — 风险调整收益恶化
    r = _get_rules(rules, "sell", "S4_risk_decay")
    if r.get("enabled", True) and sig["rolling_sharpe"] < r["sharpe_max"] and sig["monthly_return"] < r["monthly_return_max"]:
        triggers.append(
            f"S4 Risk decay (Sharpe {sig['rolling_sharpe']:.1f}, monthly {sig['monthly_return']:.1%})"
        )

    # S5: 高位回撤 — 从高点持续回落
    r = _get_rules(rules, "sell", "S5_peak_retreat")
    if r.get("enabled", True) and (sig["above_ma200"]
            and sig["pullback_from_peak"] > r["pullback_min"]
            and sig["consecutive_declines"] >= r["decline_min"]):
        triggers.append(
            f"S5 Peak retreat (pullback {sig['pullback_from_peak']:.1%}, "
            f"{sig['consecutive_declines']}d decline)"
        )

    # S6: 偏离回归 — 偏离 MA200 过多，均值回归风险
    r = _get_rules(rules, "sell", "S6_deviation")
    if r.get("enabled", True) and sig["trend_strength"] > r["trend_strength_min"] and sig["rsi_14"] > r["rsi_min"]:
        triggers.append(
            f"S6 Deviation (bias {sig['trend_strength']:.1%}, RSI={sig['rsi_14']:.0f})"
        )

    # S7: 波动飙升 — 波动率异常放大
    r = _get_rules(rules, "sell", "S7_vol_spike")
    if r.get("enabled", True) and sig["volatility_ratio"] > r["vol_ratio_min"]:
        triggers.append(
            f"S7 Vol spike ({sig['volatility_ratio']:.1f}x hist)"
        )

    # S8: 趋势确认止损 — 跌破 MA200 且月度亏损，确认下行趋势
    r = _get_rules(rules, "sell", "S8_trend_confirm")
    if r.get("enabled", True) and not sig["above_ma200"] and sig["monthly_return"] < r["monthly_return_max"]:
        triggers.append(
            f"S8 Trend confirm (bias {sig['trend_strength']:.1%}, "
            f"monthly {sig['monthly_return']:.1%})"
        )
        has_clear = True

    if triggers:
        action = "清仓" if has_clear else "减仓"
        return action, " | ".join(triggers)

    # ── Hold ───────────────────────────────
    if sig["above_ma200"]:
        return "持有", f"Uptrend (+{sig['trend_strength']:.1%} vs MA200)"
    else:
        return "持有", "Weak trend, monitoring"
