# 🌧️ RainfallBench: A Benchmark for Real-World Rainfall Time-Series Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**RainfallBench** is the first dedicated benchmark for rainfall time series forecasting using historical meteorological data. It provides a unified platform for evaluating state-of-the-art forecasting models under domain-specific challenges such as zero inflation, temporal decay, and weak periodicity.

> 📌 **If you're working on rainfall, hydro-meteorology, or time series modeling with irregular sparsity—this benchmark is designed for you.**

---
## 📈 Results Overview

<p align="center">
  <img src="pic/result.png" width="700">
</p>

**Figure:** Performance comparison of 20+ models on RainfallBench across multiple temporal scales. 

## 🔍 Motivation

Most general-purpose time series forecasting (TSF) benchmarks target finance, traffic, or power usage data. However, **rainfall forecasting** presents unique challenges:

- 🌑 **Zero-Inflation:** >70% of the data are zeros (dry periods)
- ⏳ **Temporal Decay:** Recent observations dominate signal value
- 🔄 **Weak Periodicity:** Rainfall lacks stable periodic structure
- 🌍 **Real-World Relevance:** Derived from operational meteorological stations

RainfallBench addresses these challenges by offering a **domain-specific benchmark**, a **structured evaluation strategy**, and a **plug-and-play forecasting module**.

---

## 📦 Features

- ✅ **171,000+ real-world rainfall records** (2018–2022), sampled every 15 minutes
- ✅ **Six core meteorological variables**, including `tp`, `pwv`, `rh`, `t2m`
- ✅ **Multi-resolution evaluation** across 10 input-output configurations
- ✅ **Extreme rainfall prediction** support via T/CMSA 0013-2019 standard
- ✅ **Plug-in module**: Bi-Focus Precipitation Forecaster (BFPF)

---

## 🗃️ Dataset Overview

| Variable | Description |
|----------|-------------|
| `t2m`    | Temperature at 2m above ground |
| `sp`     | Surface pressure |
| `rh`     | Relative humidity |
| `wind_speed` | Wind speed |
| `pwv`    | Precipitable water vapor (retrieved via GNSS delay inversion) |
| `tp`     | Total precipitation (target, from  GPM (IMERG) final product) |

All data are cleaned, normalized, and chronologically ordered. No missing values.

📁 The full dataset is available [here](#) (link to release or Zenodo/GitHub repo).

---

## 🧠 Model Benchmarking

We evaluate **20+ state-of-the-art models**, covering:
- 🔁 RNN-based (e.g., P-sLSTM, SegRNN)
- 🔀 Transformer-based (e.g., Informer, PatchTST, iTransformer)
- 🧠 MLP, GNN, CNN, KAN families

Each model is tested on:
- **Multi-temporal scale prediction** (input=24/48, output=4–12)
- **Extreme rainfall detection**
- **Average MSE and MAE** across all tasks

### 🎯 Our Module: Bi-Focus Precipitation Forecaster (BFPF)

We propose a plug-and-play attention enhancement module that:
- 💧 Focuses attention on **non-zero rainfall** segments
- 📍 Emphasizes **recent temporal context**
- 📈 Boosts Transformer performance, especially under sparse signals

---

### 🔧 Installation

```bash
git clone https://github.com/your-org/RainfallBench.git
cd RainfallBench
conda create -n rainfallbench python=3.10
conda activate rainfallbench
pip install -r requirements.txt
``` 

## 🙏 Acknowledgements

We appreciate the following open-source repositories for their valuable contributions to time series forecasting and attention mechanisms. Their codebases provided important foundations and insights during the development of RainfallBench and BFPF.
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)

