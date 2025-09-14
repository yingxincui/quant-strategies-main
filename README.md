# 🚀 量化交易策略回测系统

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/streamlit-1.47+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

一个功能强大的量化交易策略回测系统，基于 Python Backtrader 框架构建，使用 Streamlit 提供现代化的 Web 界面。支持多种交易策略、实时数据获取、智能风控和详细的性能分析。

> 💡 **特别说明**：本系统针对 ETF 实盘交易进行优化，推荐使用东方财富证券（免收每笔交易最低5元手续费），回测系统设置佣金万2.5以更好模拟实盘情况。

## ✨ 功能特点

### 🎨 用户界面
- 🖥️ **现代化 Web 界面**：基于 Streamlit 构建，操作简单直观
- 📈 **交互式图表**：K线图、均线、交易点位等实时展示
- 📝 **实时日志**：完整的交易记录和策略执行日志

### 📡 数据获取
- 🚀 **实时数据**：支持 Tushare 和 AKShare 双数据源
- 🌏 **多市场支持**：A股、ETF、港股等多个市场
- 🔄 **自动切换**：数据源失效时自动切换备用源

### 🧠 策略系统
- 🏭 **工厂模式**：采用策略工厂模式，新策略扩展简单
- 🔀 **多策略支持**：双均线、市场情绪、ETF轮动等多种策略
- 🧑‍💻 **智能优化**：支持强化学习和超参数优化

### ⚡ 风险管理
- 🛡️ **智能止损**：追踪止损、ATR 动态止损等
- 📉 **回撤控制**：最大回撤限制，保护资金安全
- 📊 **仓位管理**：基于 ATR 动态计算持仓规模
- 🔄 **T+1 交易**：严格遵守 T+1 交易规则

### 📊 性能分析
- 🎯 **详细指标**：夏普比率、最大回撤、胜率、盈亏比等
- 💰 **精确费用**：包含佣金、印花税等真实成本计算
- 📅 **历史分析**：完整的交易历史和收益曲线

## 功能截图
![image](https://github.com/user-attachments/assets/0af62636-dd11-44d5-8e8d-775db56df64e)

## 📦 内置策略

### 📈 双均线策略 (Dual Moving Average)
- 🎯 **信号生成**：使用快速和慢速移动平均线的交叉产生交易信号
- 🛡️ **风险控制**：支持追踪止损进行风险控制
- 📊 **仓位管理**：基于 ATR 动态计算持仓规模
- 📅 **趋势过滤**：结合53日5日均线进行趋势确认

### 💭 市场情绪策略 (Market Sentiment)
- 🌡️ **情绪指标**：基于市场情绪指标进行交易，在极端情绪时入场
- 📈 **趋势确认**：使用 EMA 趋势确认，要求价格和 EMA 同步上涨
- 🎯 **动态止盈**：动态调整 ATR 止盈倍数，结合布林带波动率和情绪因子
- 📊 **分层建仓**：在不同情绪阈值增加仓位
- 🛡️ **风控系统**：支持追踪止损和最大回撤限制

### 🔄 ETF轮动策略 (ETF Rotation)
- 💪 **动量排名**：根据双动量指标对ETF进行排名
- 🗺️ **市场状态**：智能识别市场状态，调整持仓数量
- 🔄 **定期调仓**：按设定间隔进行仓位调整
- 📉 **动量衰减**：监控动量衰减信号，及时退出弱势资产

### 🤖 强化学习策略 (RL Strategy)
- 🧠 **AI 决策**：基于深度强化学习的智能交易决策
- 📊 **多因子模型**：综合价格、成交量、技术指标等多维特征
- 🔄 **自适应学习**：模型能够从历史交易中不断学习和优化
- 🎯 **实时调整**：根据市场环境变化实时调整策略参数

## 🚀 快速开始

### 📋 系统要求

- **Python**: 3.10+
- **操作系统**: Windows / macOS / Linux
- **内存**: 推荐 8GB+
- **存储**: 至少 2GB 可用空间

### 🛠️ 安装步骤

#### 1. 克隆仓库
```bash
git clone https://github.com/sencloud/ETF-Strategies.git
cd ETF-Strategies
```

#### 2. 创建虚拟环境（推荐使用 Miniconda）
```bash
# 配置 pip 国内源（可选）
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
pip config set install.trusted-host mirrors.aliyun.com

# 创建和激活虚拟环境
conda create -n etf-strategies python=3.10
conda activate etf-strategies
```

#### 3. 安装依赖包
```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 TA-Lib（技术分析库）
conda install -c conda-forge ta-lib
```

#### 4. 启动系统
```bash
streamlit run app.py
```

系统将在浏览器中自动打开，默认地址：`http://localhost:8501`

### 🔑 数据源配置

#### Tushare 配置（推荐）
1. 注册 [Tushare](https://tushare.pro/register) 账号
2. 获取 API Token
3. 在系统界面中输入 Token

#### AKShare 配置（备用）
- 无需注册，但数据更新可能有延迟
- 系统会在 Tushare 不可用时自动切换

## 📁 项目结构

```
quant-strategies-main/
├── app.py                    # 🚀 主程序入口
├── requirements.txt          # 📦 依赖包列表
├── README.md                 # 📜 项目说明文档
├── src/                     # 💻 源代码目录
│   ├── data/                # 📡 数据加载模块
│   │   ├── data_loader.py      # A股/ETF数据加载器
│   │   └── future_data_loader.py # 期货数据加载器
│   ├── strategies/          # 📈 交易策略模块
│   │   ├── dual_ma_strategy.py           # 双均线策略
│   │   ├── market_sentiment_strategy.py  # 市场情绪策略
│   │   ├── etf_rotation_strategy.py      # ETF轮动策略
│   │   ├── rl_model_strategy.py          # 强化学习策略
│   │   ├── strategy_factory.py           # 策略工厂
│   │   └── market_sentiment/            # 市场情绪相关模块
│   ├── indicators/          # 📉 技术指标模块
│   │   └── trailing_stop.py     # 追踪止损指标
│   ├── trading/             # 💼 交易执行模块
│   │   └── market_executor.py   # 市场交易执行器
│   └── utils/               # 🔧 工具函数模块
│       ├── analysis.py          # 回测结果分析
│       ├── backtest_engine.py   # 回测引擎
│       ├── logger.py            # 日志系统
│       ├── plot.py              # 图表绘制
│       └── notification.py      # 通知系统
├── rl_model_finrl/          # 🤖 强化学习模块
│   ├── agents/              # RL智能体
│   ├── applications/        # RL应用
│   └── meta/                # 元数据处理
├── ui/                     # 🖥️ 用户界面
│   ├── pages/               # 页面组件
│   │   ├── backtest.py          # 回测页面
│   │   ├── market.py            # 实盘交易页面
│   │   ├── settings.py          # 设置页面
│   │   └── sidebar.py           # 侧边栏
│   └── articles/            # 帮助文档
└── tools/                  # 🛠️ 辅助工具
    ├── plot_json_data.py       # JSON数据可视化
    └── plot_general_json.py    # 通用JSON绘图工具
```

## 🛠️ 开发指南

### 🎯 添加新策略

#### 1. 创建策略类
在 `src/strategies/` 目录下创建新的策略类：

```python
import backtrader as bt
from loguru import logger

class YourCustomStrategy(bt.Strategy):
    params = (
        ('param1', 10),        # 策略参数 1
        ('param2', 0.02),      # 策略参数 2
        ('risk_ratio', 0.02),  # 风险比率
    )
    
    def __init__(self):
        """Initialize strategy"""
        # 初始化技术指标
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.param1)
        self.order = None
        logger.info(f"策略初始化完成 - 参数: {self.params._getpairs()}")
        
    def next(self):
        """Strategy logic"""
        if self.order:  # 检查是否有未完成订单
            return
            
        # 实现交易逻辑
        if not self.position:  # 无持仓
            if self.data.close[0] > self.sma[0]:  # 买入条件
                size = self.calculate_position_size()
                self.order = self.buy(size=size)
                logger.info(f"买入信号 - 价格: {self.data.close[0]:.2f}")
        else:  # 有持仓
            if self.data.close[0] < self.sma[0]:  # 卖出条件
                self.order = self.close()
                logger.info(f"卖出信号 - 价格: {self.data.close[0]:.2f}")
    
    def calculate_position_size(self):
        """Calculate position size based on risk"""
        cash = self.broker.getcash()
        price = self.data.close[0]
        risk_amount = self.broker.getvalue() * self.p.risk_ratio
        # 这里可以添加更复杂的仓位计算逻辑
        return int(risk_amount / price / 100) * 100  # 调整为100股的整数倍
        
    def notify_order(self, order):
        """Order notification"""
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.info(f"买入成功 - 数量: {order.executed.size}, 价格: {order.executed.price:.2f}")
            else:
                logger.info(f"卖出成功 - 数量: {order.executed.size}, 价格: {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"订单失败: {order.status}")
        
        self.order = None
```

#### 2. 注册策略
在 `src/strategies/strategy_factory.py` 中注册新策略：

```python
from .your_custom_strategy import YourCustomStrategy

# 在 StrategyFactory 类中添加
def __init__(self):
    # ... existing code ...
    self.register_strategy("您的自定义策略", YourCustomStrategy)
```

#### 3. 添加界面参数
在 `ui/pages/sidebar.py` 中添加策略参数输入：

```python
# 在 render_sidebar() 函数中添加
if strategy_name == "您的自定义策略":
    st.subheader("策略参数")
    param1 = st.number_input("参数1", value=10, min_value=1)
    param2 = st.slider("参数2", 0.01, 0.1, 0.02)
    
    # 添加到参数字典
    strategy_params.update({
        'param1': param1,
        'param2': param2,
    })
```

### 📈 技术指标开发

在 `src/indicators/` 目录下创建自定义指标：

```python
import backtrader as bt

class CustomIndicator(bt.Indicator):
    lines = ('custom_line',)  # 定义输出线
    params = (
        ('period', 14),  # 周期参数
    )
    
    def __init__(self):
        # 初始化计算
        pass
        
    def next(self):
        # 实现指标计算逻辑
        self.lines.custom_line[0] = calculate_value()
```

### 📊 数据源扩展

在 `src/data/` 目录下添加新的数据加载器：

```python
class CustomDataLoader:
    def __init__(self, api_key=None):
        self.api_key = api_key
        
    def download_data(self, symbol, start_date, end_date):
        # 实现数据下载逻辑
        pass
```

## 📊 回测指标说明

### 🎯 收益指标
- **总收益率**：策略最终收益相对于初始资金的百分比
- **年化收益率**：按252个交易日计算的年化收益
- **夏普比率**：超额收益相对于波动率的比率（>1 为优秀）

### 🛡️ 风险指标
- **最大回撤**：策略执行期间的最大亏损百分比
- **波动率加权收益(VWR)**：考虑波动率的收益指标
- **系统质量指数(SQN)**：衡量交易系统的稳定性（>2.5 为优秀）

### 💼 交易指标
- **胜率**：盈利交易占总交易次数的比例
- **盈亏比**：平均盈利交易额与平均亏损交易额的比值
- **平均持仓时间**：平均每笔交易的持仓天数

### 💰 成本分析
- **总手续费**：包含佣金、印花税等所有交易成本
- **滑点成本**：交易执行中的滑点损失
- **资金使用率**：资金的平均使用效率

## ⚠️ 风险提示

> 🚨 **重要声明**：
> 
> 1. 本系统仅供 **学习和研究** 使用，不构成任何投资建议
> 2. 使用本系统进行 **实盘交易** 需要自行承担风险
> 3. 历史数据不代表未来表现，请谨慎投资
> 4. 建议在模拟环境中充分测试后再考虑实盘应用

## ✨ 支持项目

如果你觉得这个项目对你有帮助，欢迎支持我们：

### ☕ 买杯咖啡
<img src="https://github.com/user-attachments/assets/e75ef971-ff56-41e5-88b9-317595d22f81" alt="支付二维码" width="300" height="300">

### ⭐ GitHub Star
给我们一个 Star，让更多人发现这个项目！

## 🤝 贡献指南

欢迎各种形式的贡献！

### 🐛 报告 Bug
- 使用 [GitHub Issues](https://github.com/sencloud/ETF-Strategies/issues) 报告问题
- 请提供详细的错误信息和复现步骤

### 🚀 功能建议
- 通过 Issues 提出新功能建议
- 详组描述需求和使用场景

### 📝 代码贡献
1. Fork 本仓库
2. 创建特性分支: `git checkout -b feature/AmazingFeature`
3. 提交修改: `git commit -m 'Add some AmazingFeature'`
4. 推送分支: `git push origin feature/AmazingFeature`
5. 提交 Pull Request

### 📚 文档改进
- 改进代码注释和文档
- 翻译文档到其他语言
- 添加使用示例和教程

## 📜 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

<div align="center">

**🎆 感谢使用 ETF 量化交易系统！🎆**

如有问题或建议，欢迎 [Issues](https://github.com/sencloud/ETF-Strategies/issues) 交流讨论。

[![GitHub Stars](https://img.shields.io/github/stars/sencloud/ETF-Strategies?style=social)](https://github.com/sencloud/ETF-Strategies/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/sencloud/ETF-Strategies?style=social)](https://github.com/sencloud/ETF-Strategies/network/members)

</div>
