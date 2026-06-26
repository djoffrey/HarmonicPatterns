# HarmonicPatterns

A modern, high-performance Python library for detecting and predicting **Harmonic Trading Patterns** in financial charts. This library filters ZigZag movements and checks if their relationships (retracements/projections) align with standard Fibonacci ratios.

---

## 🌟 Features

- **9 Supported Patterns**: Detects ABCD, Gartley, Bat, Alt-Bat, Butterfly, Crab, Deep Crab, Shark, and Cypher.
- **Both Completed & Predicting Scanning**: Not only matches historical patterns but also calculates potential D-point targets for incomplete patterns (predicting).
- **Visualization**: Seamlessly plots K-lines, moving averages, and detected pattern shapes using `mplfinance` and saves them to images.
- **Modern Package Management**: Powered by `uv` for reproducible builds and lightning-fast package resolution.

---

## 🛠️ Installation

### 1. Prerequisites (TA-Lib C Library)

This library depends on the python wrapper `TA-Lib`, which requires the underlying C library to be installed on your machine.

#### macOS
```bash
brew install ta-lib
```

#### Linux (Debian/Ubuntu)
Download and compile the source:
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

---

### 2. Setting Up the Environment with `uv`

We recommend using `uv` (a fast alternative to pip/virtualenv) to manage the virtual environment and dependencies.

First, install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Initialize the virtual environment and install the package in editable mode:
```bash
# Create a virtual environment
uv venv

# Lock dependencies and install in editable mode (including requirements)
uv lock
uv pip install -e .
```

---

## 🚀 Quickstart

Here is a simple example to fetch BTC/USDT data and scan for patterns:

```python
import ccxt
from harmonic_patterns import HarmonicDetector, kline_to_df

# 1. Initialize exchange & fetch data
exchange = ccxt.binance()
raw_candles = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=200)
df = kline_to_df(raw_candles)

# 2. Instantiate the detector
# error_allowed=0.05 allows a 5% deviation from standard ratios
detector = HarmonicDetector(error_allowed=0.05, strict=True)

# 3. Detect patterns
# return_dt_idx=True returns timestamps instead of integer index points
patterns, predict_patterns = detector.search_patterns(
    df, 
    only_last=False, 
    last_n=4, 
    plot=True, 
    predict=True,
    return_dt_idx=True,
    save_fig_name='btc_patterns.png'
)

# 4. View results
print(f"Completed patterns: {len(patterns)}")
print(f"Predicting patterns: {len(predict_patterns)}")
```

---

## 📊 Supported Patterns & Fibonacci Ratios

| Pattern | XA to AB (XAB) | AB to BC (ABC) | BC to CD (BCD) | XA to AD (XAD) / XC to CD (XCD) |
| :--- | :--- | :--- | :--- | :--- |
| **ABCD** | - | $0.382 \sim 0.886$ | $1.13 \sim 1.168$ | $AB = CD$ ($BCD / (1/ABC) = 1$) |
| **Gartley** | $0.618$ | $0.382 \sim 0.886$ | $1.13 \sim 1.168$ | $XAD = 0.786$ |
| **Bat** | $0.382 \sim 0.5$ | $0.382 \sim 0.886$ | $1.618 \sim 2.168$ | $XAD = 0.886$ |
| **Alt-Bat** | $0.382$ | $0.382 \sim 0.886$ | $2.0 \sim 3.168$ | $XAD = 1.13$ |
| **Butterfly** | $0.786$ | $0.382 \sim 0.886$ | $1.618 \sim 2.24$ | $XAD = 1.27$ |
| **Crab** | $0.382 \sim 0.618$ | $0.382 \sim 0.886$ | $2.618 \sim 3.618$ | $XAD = 1.618$ |
| **Deep Crab** | $0.886$ | $0.382 \sim 0.886$ | $2.24 \sim 3.618$ | $XAD = 1.618$ |
| **Shark** | $0.5 \sim 0.886$ | $1.13 \sim 1.618$ | $1.618 \sim 2.24$ | $XAD = 0.886 \sim 1.13$ |
| **Cypher** | $0.382 \sim 0.786$ | $1.272 \sim 1.414$ | - | $XCD = 0.786$ or $XAD = 0.786$ |

---

## 🏃 Running Examples

We provide a complete, real-time scanning example using OKX market data:

```bash
# Run real-time OKX scanning for BTC & ETH
python examples/run_okx_example.py
```

This will output scan results directly to the terminal, and save visual charts (K-lines with harmonic lines) under the `./data` folder.

---

## 📂 Project Structure

- `src/`: Core library code.
  - [harmonic_patterns.py](src/harmonic_patterns.py): Core detector logic, ZigZag algorithms, and plot functions.
  - [harmonic_functions.py](src/harmonic_functions.py): Wrapper module matching legacy imports.
  - [settings.py](src/settings.py): Scanning variables and default parameters.
- `examples/`: Handy runnable scripts and notebooks.
  - [run_okx_example.py](examples/run_okx_example.py): Real-time scanner for OKX.
  - [run_detect.py](examples/run_detect.py): Multi-threaded polling scanning script.
- `pyproject.toml`: Modern PEP-518 project definition.
