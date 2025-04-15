# HKStock-Quant

一个高可用、可扩展的港股量化交易系统，支持数据获取、策略开发、回测、可视化和实盘交易。

## 项目结构

```
hkstock-quant/
├── src/                    # 源代码
│   ├── data/               # 数据获取和处理模块
│   ├── strategies/         # 交易策略
│   ├── models/             # 预测模型
│   ├── backtest/           # 回测引擎
│   ├── visualization/      # 可视化工具
│   ├── trading/            # 交易执行模块
│   ├── utils/              # 通用工具函数
│   └── config/             # 配置文件
├── tests/                  # 单元测试
├── docs/                   # 文档
├── notebooks/              # Jupyter笔记本，用于研究和分析
├── data/                   # 数据存储
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后的数据
├── logs/                   # 日志文件
├── requirements.txt        # 项目依赖
├── pyproject.toml          # 项目配置和依赖管理
└── .env.example            # 环境变量示例文件
```

## 功能特点

- **数据获取**：支持从多个数据源获取港股市场数据
- **策略开发**：模块化的策略开发框架
- **回测系统**：基于 backtesting.py 和 backtrader 的回测引擎
- **可视化分析**：交易结果和性能指标的可视化
- **实盘交易**：支持通过富途证券等接口进行实盘交易
- **风险管理**：内置风险控制模块

## 安装

### 使用 uv 进行安装（推荐）

1. 克隆仓库
```bash
git clone https://github.com/yourusername/hkstock-quant.git
cd hkstock-quant
```

2. 运行初始化脚本
```bash
python init_uv.py
```

3. 激活虚拟环境
```bash
# 在 macOS/Linux 上
source .venv/bin/activate

# 在 Windows 上
.venv\Scripts\activate
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件，填入API密钥等敏感信息
```

### 使用 pip 进行安装

1. 克隆仓库
```bash
git clone https://github.com/yourusername/hkstock-quant.git
cd hkstock-quant
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件，填入API密钥等敏感信息
```

## 使用示例

### 回测策略

```python
from src.backtest.engine import BacktestEngine
from src.strategies.moving_average import MovingAverageStrategy

# 初始化回测引擎
engine = BacktestEngine(
    strategy=MovingAverageStrategy,
    start_date='2022-01-01',
    end_date='2023-01-01',
    symbols=['0700.HK', '9988.HK'],
    initial_capital=100000
)

# 运行回测
results = engine.run()

# 可视化结果
engine.visualize()
```

### 实盘交易

```python
from src.trading.futu_trader import FutuTrader
from src.strategies.moving_average import MovingAverageStrategy

# 初始化交易者
trader = FutuTrader(
    strategy=MovingAverageStrategy,
    symbols=['0700.HK', '9988.HK'],
    config_path='src/config/futu_config.yaml'
)

# 启动交易
trader.start()
```

## 开发指南

### 项目架构说明

本项目采用模块化设计，各个组件之间低耦合高内聚，便于扩展和维护。主要模块包括：

1. **数据模块 (`src/data/`)**：
   - `data_fetcher.py`：负责从各种数据源获取数据，支持多种数据源（AKShare、YFinance等）
   - 实现了数据缓存机制，避免重复请求
   - 支持数据预处理和技术指标计算

2. **策略模块 (`src/strategies/`)**：
   - `base_strategy.py`：策略基类，定义了策略接口
   - `moving_average.py`：移动平均策略示例
   - 策略模块与数据和执行模块解耦，便于独立开发和测试

3. **回测模块 (`src/backtest/`)**：
   - `engine.py`：回测引擎，支持多股票回测、交易成本、滑点等
   - 提供完整的性能指标计算（夏普比率、最大回撤、胜率等）
   - 支持策略参数优化

4. **可视化模块 (`src/visualization/`)**：
   - `visualizer.py`：提供多种可视化工具，支持静态和交互式图表
   - 支持生成完整的回测报告和仪表盘

5. **交易模块 (`src/trading/`)**：
   - `base_trader.py`：交易基类，定义了交易接口
   - `futu_trader.py`：富途API交易实现
   - 支持订单管理、仓位跟踪和风险控制

6. **工具模块 (`src/utils/`)**：
   - `logger.py`：日志工具，支持控制台和文件日志
   - 各种辅助功能

7. **配置模块 (`src/config/`)**：
   - `config.py`：配置管理，支持环境变量和配置文件
   - 不同环境的配置分离

### 开发注意事项

1. **环境配置**：
   - 使用 uv 管理依赖，确保环境一致性
   - 敏感信息（API密钥等）存放在 `.env` 文件中，不要提交到版本控制系统
   - 使用 `pyproject.toml` 管理项目依赖，便于版本控制

2. **代码规范**：
   - 遵循 PEP 8 编码规范
   - 使用类型注解提高代码可读性和可维护性
   - 编写详细的文档字符串，说明函数/类的用途、参数和返回值
   - 使用 black、isort、flake8 和 mypy 进行代码格式化和静态检查

3. **测试**：
   - 为核心功能编写单元测试
   - 使用 pytest 运行测试
   - 测试数据与生产数据分离

4. **数据处理**：
   - 实现数据缓存机制，避免频繁请求数据源
   - 处理缺失数据和异常值
   - 数据预处理应该是可重复的

5. **策略开发**：
   - 继承 `BaseStrategy` 类开发新策略
   - 实现 `generate_signals` 方法生成交易信号
   - 策略参数应该是可配置的
   - 避免过度拟合历史数据

6. **回测**：
   - 考虑交易成本、滑点和流动性
   - 使用多个时间段和多个股票进行回测
   - 进行参数敏感性分析
   - 注意前视偏差（look-ahead bias）

7. **实盘交易**：
   - 实盘前充分测试策略
   - 实现风险控制机制（止损、头寸管理等）
   - 监控系统状态和交易执行
   - 实现异常处理和恢复机制

8. **性能优化**：
   - 对计算密集型操作进行优化
   - 考虑使用并行计算处理大量数据
   - 优化数据存储和访问

9. **扩展性**：
   - 添加新数据源：扩展 `DataFetcher` 类
   - 添加新策略：继承 `BaseStrategy` 类
   - 添加新交易接口：继承 `BaseTrader` 类
   - 添加新可视化工具：扩展 `Visualizer` 类

### 常见问题解决

1. **数据获取问题**：
   - 检查网络连接和API密钥
   - 查看日志文件了解详细错误信息
   - 尝试更换数据源

2. **回测性能问题**：
   - 减少回测时间范围或股票数量
   - 优化策略计算逻辑
   - 使用缓存数据

3. **实盘交易问题**：
   - 确保富途OpenD已启动并正确配置
   - 检查账户余额和交易权限
   - 查看日志文件了解详细错误信息

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议。请遵循以下步骤：

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

[MIT](LICENSE)
