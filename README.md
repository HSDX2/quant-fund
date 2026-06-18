# fund-metrics

A股公募基金量化分析系统。每日自动筛选基金池、获取净值与盘中估值、计算技术指标、规则匹配买卖信号、生成持仓建议、推送微信报告。

## 特性

- **全市场覆盖**：从 27,000+ 只公募基金中筛选，支持股票型、混合型、债券型、QDII、ETF 等
- **规则驱动**：8 条买入规则 + 8 条卖出规则，逻辑透明可解释
- **学术理论支撑**：仓位计算基于 Half-Kelly、Vol Targeting（Moreira & Muir 2017）、Drawdown Constraint（Grossman & Zhou 1993）
- **盘中估值**：集成天天基金实时估值，盘前即可预判当日走势
- **微信推送**：通过 PushPlus 每日自动推送完整报告
- **基金池缓存**：首次筛选后缓存，后续秒级加载

## 快速开始

### 环境要求

- Python 3.8+
- Windows / Linux / macOS
- 网络可访问东方财富 API 和天天基金

### 安装

```bash
git clone https://github.com/SuperBigFive/quant-fund.git
cd quant-fund
pip install -r requirements.txt
```

### 配置

```bash
# 1. 创建环境变量文件
echo "PUSHPLUS_TOKEN=你的token" > .env

# 2. 编辑持仓列表（每行一个 6 位基金代码）
# 编辑 holdings.txt
```

PushPlus token 在 [pushplus.plus](http://www.pushplus.plus) 免费获取。

### 运行

```bash
python main.py                     # 完整流程（估值 + 推送）
python main.py --no-estimate       # 跳过盘中估值（盘后使用）
python main.py --no-push           # 不推送微信
python main.py --refresh-universe  # 强制重建基金池
python main.py --fund 000001       # 单只基金诊断
```

## 流水线架构

```
                 ┌─────────────────┐
                 │  1. 基金池筛选   │  universe.py
                 │  类型/关键词过滤  │  规模≥5亿 成立≥1年
                 │  同类去重        │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │  2. 净值获取     │  data_fetcher.py
                 │  东方财富 API    │  串行拉取 + 日缓存
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │  2.5 盘中估值    │  天天基金 fundgz
                 │  追加今日数据点   │  并行获取 + 重试
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │  3. 技术指标     │  indicators.py
                 │  MA200/RSI/夏普  │  异常基金自动排除
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │  4. 规则匹配     │  scorer.py
                 │  8买 + 8卖规则   │  买入 TOP 10
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │  5. 仓位计算     │  position_advisor.py
                 │  Kelly + Vol + DD│  单只≤5% 总仓≤80%
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │  6. 报告推送     │  reporter.py
                 │  PushPlus → 微信 │  持仓建议 + 买入推荐
                 └─────────────────┘
```

## 买卖规则

### 买入（命中任一即入围，按反弹潜力排序取 TOP 10）

| 规则 | 条件 |
|------|------|
| P1 黄金回调 | MA200 上方 ∧ 连跌 3-5 天 ∧ RSI < 40 |
| P2 超跌反弹 | MA200 上方 ∧ RSI < 30 ∧ 60 日回撤 > 8% |
| P3 趋势回调 | MA200 上方 ∧ 连跌 3-5 天 ∧ 夏普 > 0.5 |
| P4 低波抄底 | MA200 上方 ∧ RSI < 35 ∧ 波动率 < 历史均值 |
| P5 深度价值 | MA200 上方 ∧ 60 日回撤 > 12% ∧ RSI < 35 |
| P6 优质回调 | MA200 上方 ∧ 夏普 > 1.0 ∧ 连跌 2-5 天 |
| P7 波动收缩 | MA200 上方 ∧ 波动率 < 0.6× ∧ 高点回撤 > 3% |
| P8 强趋势回调 | MA200 上方 ∧ 趋势强度 > 5% ∧ 连跌 ≥ 2 天 |

### 卖出（命中任一即触发）

| 规则 | 条件 | 操作 |
|------|------|------|
| S1 趋势反转 | 跌破 MA200 | 清仓 |
| S2 加速恶化 | 60 日回撤 > 10% ∧ 波动率 > 1.5× | 减仓 |
| S3 过热止盈 | 连涨 ≥ 5 天 ∧ RSI > 70 | 减仓 |
| S4 风险恶化 | 夏普 < 0 ∧ 月度亏损 < −5% | 减仓 |
| S5 高位回撤 | MA200 上方 ∧ 高点回撤 > 5% ∧ 连跌 ≥ 4 天 | 减仓 |
| S6 质量崩塌 | 夏普 < −1.0 | 清仓 |
| S7 波动爆炸 | 波动率 > 2.0× 历史 | 减仓 |
| S8 持续阴跌 | MA200 上方 ∧ 连跌 ≥ 6 天 | 减仓 |

### 仓位计算

| 场景 | 方法 | 理论来源 |
|------|------|----------|
| 买入 | Half-Kelly × 波动率调整 × 回撤缩放 | Kelly (1956), Moreira & Muir (2017) |
| 卖出 | max(Vol Targeting, DD Constraint) | Moreira & Muir (2017), Grossman & Zhou (1993) |
| 约束 | 单只 ≤ 5%，总仓 ≤ 80% | — |

## 技术指标

每只基金计算 14 个指标：

| 类别 | 指标 | 说明 |
|------|------|------|
| 趋势 | MA200、趋势强度、是否站上 MA200 | 200 日均线系统 |
| 动量 | RSI(14)、单日涨跌、近月收益 | 相对强弱与短期收益 |
| 回调 | 连跌/连涨天数、累计跌幅、高点回撤 | 近期调整幅度 |
| 风险 | 波动率比、60 日最大回撤、滚动夏普 | 风险度量 |
| 估值 | 盘中实时估值（天天基金） | 当日预判 |

## 定时运行

建议配置 cron，每个交易日收盘前 15 分钟自动运行：

```bash
# 编辑 crontab
crontab -e

# 添加（仅交易日）
45 14 * * 1-5 cd /path/to/quant-fund && python main.py
```

## 配置说明

编辑 `config.yaml`：

```yaml
universe:
  min_aum_yi: 5             # 排除规模 < 5 亿的小基金
  min_fund_age_days: 365    # 排除成立 < 1 年的新基金
  include_types:             # 纳入的基金类型
    - "股票指数"
    - "混合型"
    - "债券型"
    - "QDII"
    # ...更多类型
  exclude_keywords:          # 排除的行业/主题
    - "医药"
    - "军工"
    # ...更多关键词

cache:
  universe_refresh_days: 1   # 基金池缓存过期天数
  nav_history_days: 500      # 净值拉取历史天数
```

## 文件结构

```
quant-fund/
├── main.py               # 主入口 + CLI
├── config.yaml           # 策略与筛选参数
├── requirements.txt      # Python 依赖
├── universe.py           # 基金池筛选（类型/规模/Age/去重）
├── data_fetcher.py       # 数据获取（净值 + 估值 + 基本信息）
├── indicators.py         # 技术指标计算
├── scorer.py             # 买卖规则匹配
├── holdings.py           # 持仓分析
├── position_advisor.py   # 仓位计算（Kelly + Vol + DD）
├── reporter.py           # 报告生成
├── notifier.py           # PushPlus 微信推送
├── estimate.py           # 独立估值工具
├── .env                  # 密钥（不入版本控制）
├── holdings.txt          # 持仓代码（不入版本控制）
└── cache/                # 净值缓存 + 基金池缓存（不入版本控制）
```

## 注意事项

- 首次运行需拉取全量基金数据 + 基本信息，耗时约 5-8 分钟。后续通过缓存秒级启动
- 债券型和部分 QDII 无盘中估值，报告自动回退到前日净值
- 东方财富 API 对并发敏感，净值拉取为串行模式（约 0.3s/只），全量约 15-20 分钟
- 基金池每日自动刷新，也可 `--refresh-universe` 手动触发

## License

MIT