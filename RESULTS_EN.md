# Research Results

**Author:** Iurii Krotov
**Date:** January 2026

---

## 1. Experiment Parameters

| Parameter | Value |
|-----------|-------|
| Data period | 2010-01-01 — 2024-12-31 |
| Number of stocks | 20 (from 10 S&P 500 sectors) |
| Training window | 1260 days (5 years) |
| Forecast horizon | 21 days (1 month) |
| Risk-free rate | 4% annually |
| Number of periods | 119 |
| Covariance matrix | Ledoit-Wolf |
| Weight constraints | min=1%, max=20%, long-only |

### PatchTST Parameters (full mode)

| Parameter | Value |
|-----------|-------|
| input_length | 252 (1 year) |
| pred_length | 21 |
| patch_length | 16 |
| stride | 8 |
| d_model | 128 |
| n_heads | 16 |
| n_layers | 3 |
| d_ff | 512 |
| dropout | 0.1 |
| mask_ratio | 0.15 |
| pretrain_epochs | 20 |
| finetune_epochs | 10 |
| learning_rate | 0.005 |

---

## 2. Portfolio Metrics Comparison

| Metric | Baseline 1 | StatsForecast | PatchTST | Best |
|--------|------------|---------------|----------|------|
| **Annual Return** | **16.08%** | 13.26% | 14.07% | Baseline 1 |
| **Annual Volatility** | 12.83% | 13.26% | **12.65%** | PatchTST |
| **Sharpe Ratio** | **0.93** | 0.71 | 0.80 | Baseline 1 |
| **Calmar Ratio** | 0.66 | 0.64 | **0.90** | PatchTST |
| **Max Drawdown** | -24.22% | -20.72% | **-15.61%** | PatchTST |
| **Total Return** | **338.71%** | 243.62% | 268.82% | Baseline 1 |

### Results Interpretation

1. **By return:** Baseline 1 (historical mean) showed the highest return — 16.08% annually and 338.71% over the entire period. PatchTST came second with 14.07% annually.

2. **By risk:** PatchTST demonstrated **significantly better** risk management:
   - Minimum volatility: 12.65%
   - **Minimum drawdown: -15.61%** (36% better than Baseline 1)

3. **By risk/return ratio:**
   - Sharpe Ratio: Baseline 1 leads (0.93), but PatchTST is close (0.80)
   - **Calmar Ratio: PatchTST leads (0.90 vs 0.66)** — 36% better than Baseline 1

---

## 3. Forecast Quality Metrics

| Metric | Baseline 1 | StatsForecast | PatchTST |
|--------|------------|---------------|----------|
| **RMSE** | **0.0695** | 0.0698 | 0.0845 |
| **MAE** | **0.0506** | 0.0510 | 0.0621 |
| **Hit Rate** | **56.13%** | 52.62% | 50.88% |

### Interpretation

- Baseline 1 has the best forecast metrics by accuracy
- Hit Rate (correct direction proportion) is slightly above 50% — financial series are difficult to forecast
- PatchTST has worse forecast metrics, but **better portfolio risk metrics** — this shows that forecast accuracy is not the only factor in portfolio success

---

## 4. Key Findings

### 4.1. Main Result: PatchTST is Best for Risk Management

**Original hypothesis:** Replacing historical means with PatchTST forecasts will improve Markowitz portfolio quality.

**Result:** The hypothesis **was partially confirmed**:
- By Sharpe Ratio: Baseline 1 leads (0.93 vs 0.80)
- **By Calmar Ratio: PatchTST leads (0.90 vs 0.66)** — 36% improvement
- **By Max Drawdown: PatchTST leads (-15.61% vs -24.22%)** — 36% improvement

### 4.2. PatchTST Advantages

PatchTST showed **significant advantages** in risk management:

| Risk Metric | Baseline 1 | PatchTST | Improvement |
|-------------|------------|----------|-------------|
| Max Drawdown | -24.22% | -15.61% | **+36%** |
| Volatility | 12.83% | 12.65% | +1% |
| Calmar Ratio | 0.66 | 0.90 | **+36%** |

**Conclusion:** PatchTST forms portfolios with **significantly lower drawdowns** at comparable returns.

### 4.3. Why Does PatchTST Manage Risk Better?

1. **Adaptability to market regimes**
   - PatchTST learns to recognize patterns preceding crises
   - The model reduces allocation to risky assets before downturns

2. **Self-Supervised learning**
   - Patch masking teaches the model to understand time series structure
   - This improves volatility forecasting, not just direction

3. **Transformer architecture**
   - Attention mechanism allows capturing long-term dependencies
   - The model can "see" warning signals earlier

### 4.4. Empirical Portfolio Weight Analysis

To validate the hypotheses about better risk management, a detailed analysis of portfolio weights was conducted.

#### Behavior During Drawdown Periods

In the top-5 worst periods for Baseline 1, PatchTST showed systematically better results:

| Date | Baseline 1 | PatchTST | Difference |
|------|------------|----------|------------|
| 2022-04-08 | -10.98% | -8.40% | +2.58% |
| 2020-03-11 | -3.71% | +0.76% | +4.47% |
| 2022-09-09 | -8.72% | -6.52% | +2.20% |
| 2022-05-09 | -7.94% | -5.85% | +2.09% |
| 2022-06-09 | -7.48% | -5.27% | +2.21% |

#### Example: Worst Period (2022-04-08)

**PatchTST increased allocation to defensive assets:**
- CVX (Energy): +19.0%
- PFE (Healthcare): +18.1%
- PG (Consumer Staples): +11.9%

**PatchTST decreased allocation to volatile assets:**
- AAPL (Technology): -19.0%
- MSFT (Technology): -19.0%
- UNH (Healthcare): -11.0%

#### Average Weight Differences

PatchTST systematically holds **less** of volatile growth stocks:

| Asset | Sector | Difference (PT - B1) |
|-------|--------|----------------------|
| UNH | Healthcare | -11.2% |
| MSFT | Technology | -9.2% |
| HD | Consumer Disc. | -6.5% |
| AAPL | Technology | -5.8% |

PatchTST systematically holds **more** of defensive assets:

| Asset | Sector | Difference (PT - B1) |
|-------|--------|----------------------|
| INTC | Technology | +3.8% |
| KO | Consumer Staples | +3.4% |
| JPM | Financials | +3.0% |
| XOM | Energy | +2.8% |

#### Rebalancing Frequency

| Model | Average Weight Volatility | Ratio |
|-------|--------------------------|-------|
| Baseline 1 | 4.23% | 1.00x |
| PatchTST | 7.13% | **1.69x** |

**Conclusion:** PatchTST changes weights 1.69 times more frequently, allowing faster response to changing market conditions.

---

## 5. Practical Implications

| Investor Goal | Recommended Approach | Rationale |
|---------------|---------------------|-----------|
| Maximum return | Baseline 1 | Sharpe 0.93, Return 16.08% |
| **Minimum drawdown** | **PatchTST** | Max DD -15.61% vs -24.22% |
| **Best Calmar** | **PatchTST** | 0.90 vs 0.66 (+36%) |
| Conservative strategy | PatchTST | Better risk management |

**For institutional investors** with drawdown constraints, PatchTST is the **preferred choice**.

**For individual investors** with long-term horizons, Baseline 1 provides higher returns.

---

## 6. Summary Results Table

```
============================================================
PORTFOLIO METRICS (119 periods, 2015-2024)
============================================================

Metric                     Baseline 1      StatsF    PatchTST
-------------------------------------------------------------
Annual Return                  16.08%      13.26%      14.07%
Annual Volatility              12.83%      13.26%      12.65%
Sharpe Ratio                     0.93        0.71        0.80
Calmar Ratio                     0.66        0.64        0.90  ★
Max Drawdown                  -24.22%     -20.72%     -15.61% ★
Total Return                  338.71%     243.62%     268.82%

★ = best result for risk management

============================================================
FORECAST METRICS
============================================================

Metric                     Baseline 1      StatsF    PatchTST
-------------------------------------------------------------
RMSE                         0.069522    0.069795    0.084494
MAE                          0.050634    0.051044    0.062141
Hit Rate                       56.13%      52.62%      50.88%
```

---

## 7. Recommendations for Further Research

1. **Improve PatchTST Sharpe Ratio**
   - ~~Combine with Baseline 1 (ensemble)~~ — **tested, does not improve results** (models optimize opposite objectives: Sharpe vs Calmar)
   - Add regularization on forecast volatility

2. **Explore other markets**
   - Cryptocurrencies (high volatility)
   - Emerging markets

3. **Add additional features**
   - VIX (volatility index)
   - Macroeconomic data

---

## Conclusion

The study showed that **PatchTST is the best choice for risk-oriented portfolio management**:

- **Calmar Ratio 0.90** — 36% better than the classical approach (0.66)
- **Max Drawdown -15.61%** — 36% better than Baseline 1 (-24.22%)
- **Sharpe Ratio 0.80** — close to the leader (0.93)

For investors prioritizing **capital protection** over return maximization, PatchTST is the preferred method for estimating expected returns in Markowitz portfolio optimization.
