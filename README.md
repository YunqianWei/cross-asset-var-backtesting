@'

\# Cross-Asset Volatility Forecasting and VaR Backtesting



This project compares traditional statistical volatility models and machine learning volatility models for one-day Value-at-Risk (VaR) forecasting across US ETFs, European equity indices, and major cryptocurrencies.



\## Research Question



Can traditional statistical models and machine learning models improve VaR forecasting across different asset classes, especially during high-volatility market regimes?



\## Asset Universe



\### US ETFs

\- SPY: S\&P 500 ETF

\- QQQ: Nasdaq 100 ETF

\- IWM: Russell 2000 ETF

\- TLT: Long-term US Treasury ETF

\- GLD: Gold ETF



\### European Equity Indices

\- FTSE 100

\- DAX

\- CAC 40

\- Euro Stoxx 50



\### Cryptocurrencies

\- BTC-USD

\- ETH-USD

\- SOL-USD

\- DOGE-USD

\- XRP-USD



\## Methodology



The project uses daily close-to-close log returns and compares:



\- Historical volatility

\- EWMA volatility

\- GARCH(1,1) with Normal errors

\- GARCH(1,1) with Student-t errors

\- Ridge Regression

\- Random Forest

\- XGBoost or LightGBM



The models are evaluated using:



\- 95% and 99% one-day VaR violation rates

\- Kupiec unconditional coverage test

\- Average exceedance size

\- High-volatility regime performance

\- Traditional asset vs cryptocurrency comparison



\## Project Structure



```text

data/       Raw and processed data, not uploaded to GitHub

notebooks/  Exploratory analysis and model experiments

src/        Reusable Python modules

figures/    Generated plots

reports/    Final report and paper-style write-up

