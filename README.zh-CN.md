# HarmonicPatterns

一个用于在金融图表中检测与预测**谐波交易模式 (Harmonic Trading Patterns)** 的现代化、高性能 Python 库。该库通过提取 ZigZag 拐点，并检索其比例关系（回撤与投影）是否符合标准的斐波那契比例来进行模式匹配。

---

## 🌟 项目特性

- **支持 9 种谐波模式**：包含 ABCD, Gartley, Bat, Alt-Bat, Butterfly, Crab, Deep Crab, Shark, 以及 Cypher。
- **支持已完成模式检测与预测模式扫描**：不仅能匹配历史已经形成的完整模式，还能预测未完成模式的 D 点价格目标（预测中模式）。
- **专业可视化**：结合 `mplfinance` 和 `matplotlib`，支持在 K 线图上自动绘制移动平均线（MA）和谐波几何图形，并支持保存为本地图片。
- **现代化依赖管理**：采用 `uv` 工具，实现依赖关系的毫秒级解析与锁定。

---

## 🛠️ 安装环境部署

### 1. 安装 TA-Lib C 语言库

本库底层依赖 `TA-Lib` 库，在安装 Python 封装包之前需要先在系统中部署其 C 语言库。

#### macOS
```bash
brew install ta-lib
```

#### Linux (Debian/Ubuntu)
下载源码并编译安装：
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

---

### 2. 使用 `uv` 初始化 Python 虚拟环境

推荐使用 `uv` 管理虚拟环境和依赖包（更加安全、高效）。

安装 `uv`（如果尚未安装）：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

在项目根目录下初始化环境并以可编辑模式（Editable Mode）安装本项目：
```bash
# 创建虚拟环境
uv venv

# 锁定依赖并安装项目
uv lock
uv pip install -e .
```

---

## 🚀 极简快速上手

以下是一个使用币安 (Binance) 数据源扫描 BTC/USDT 一小时线谐波模式的极简代码：

```python
import ccxt
from harmonic_patterns import HarmonicDetector, kline_to_df

# 1. 初始化交易所并获取 K 线数据
exchange = ccxt.binance()
raw_candles = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=200)
df = kline_to_df(raw_candles)

# 2. 实例化检测器
# error_allowed=0.05 代表允许 5% 的比例误差；strict=True 要求基准必须为斐波那契数列
detector = HarmonicDetector(error_allowed=0.05, strict=True)

# 3. 执行扫描检测
# return_dt_idx=True 会将返回的节点索引转换成具体的 Timestamp 时间对象
patterns, predict_patterns = detector.search_patterns(
    df, 
    only_last=False, 
    last_n=4, 
    plot=True, 
    predict=True,
    return_dt_idx=True,
    save_fig_name='btc_patterns.png'
)

# 4. 查看结果
print(f"已形成模式数量: {len(patterns)}")
print(f"预测中模式数量: {len(predict_patterns)}")
```

---

## 📊 支持的谐波模式与斐波那契回撤比例表

| 模式名称 | XA 对 AB (XAB) | AB 对 BC (ABC) | BC 对 CD (BCD) | XA 对 AD (XAD) / XC 对 CD (XCD) |
| :--- | :--- | :--- | :--- | :--- |
| **ABCD** | - | $0.382 \sim 0.886$ | $1.13 \sim 1.168$ | $AB = CD$ ($BCD / (1/ABC) = 1$) |
| **Gartley** | $0.618$ | $0.382 \sim 0.886$ | $1.13 \sim 1.168$ | $XAD = 0.786$ |
| **Bat** | $0.382 \sim 0.5$ | $0.382 \sim 0.886$ | $1.618 \sim 2.168$ | $XAD = 0.886$ |
| **Alt-Bat** | $0.382$ | $0.382 \sim 0.886$ | $2.0 \sim 3.168$ | $XAD = 1.13$ |
| **Butterfly** | $0.786$ | $0.382 \sim 0.886$ | $1.618 \sim 2.24$ | $XAD = 1.27$ |
| **Crab** | $0.382 \sim 0.618$ | $0.382 \sim 0.886$ | $2.618 \sim 3.618$ | $XAD = 1.618$ |
| **Deep Crab** | $0.886$ | $0.382 \sim 0.886$ | $2.24 \sim 3.618$ | $XAD = 1.618$ |
| **Shark** | $0.5 \sim 0.886$ | $1.13 \sim 1.618$ | $1.618 \sim 2.24$ | $XAD = 0.886 \sim 1.13$ |
| **Cypher** | $0.382 \sim 0.786$ | $1.272 \sim 1.414$ | - | $XCD = 0.786$ 或 $XAD = 0.786$ |

---

## 🏃 运行示例脚本

我们提供了一个连接 OKX 行情的实时扫描及图表绘制示例文件：

```bash
# 运行 OKX 主流币及平台币实时扫描
python examples/run_okx_example.py
```

该脚本将实时在终端输出扫描结果，若发现模式，会将带有关联线条的图表自动保存至本地 `./data` 文件夹下。

---

## 📂 项目结构说明

- `src/`：核心库源码文件夹
  - [harmonic_patterns.py](src/harmonic_patterns.py)：核心检测器逻辑、ZigZag 拐点搜索与图表绘制函数。
  - [harmonic_functions.py](src/harmonic_functions.py)：历史导入存根，提供 100% 向后兼容。
  - [settings.py](src/settings.py)：扫描服务的配置参数与默认主流币。
- `examples/`：可执行实例和 Jupyter 记事本
  - [run_okx_example.py](examples/run_okx_example.py)：OKX 实时行情扫描绘制示例。
  - [run_detect.py](examples/run_detect.py)：多进程轮询扫描脚本。
- `pyproject.toml`：遵循现代 PEP-518 标准的库配置文件。
