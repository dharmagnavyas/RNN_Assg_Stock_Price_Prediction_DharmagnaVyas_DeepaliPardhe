# Stock Price Prediction Using RNNs

A machine learning project that implements and compares Simple RNN and Advanced RNN (GRU) models for predicting stock prices of major technology companies.

## Project Overview

This project analyzes historical stock data from four major technology companies (Amazon, Google, IBM, and Microsoft) to predict future stock prices using recurrent neural networks. The analysis covers data from 2006 to 2018 and implements comprehensive hyperparameter tuning to compare different RNN architectures.

## Objectives

- Predict stock closing prices using historical OHLCV data
- Compare performance between Simple RNN and Advanced RNN models
- Implement proper time series preprocessing techniques
- Analyze patterns in financial data across multiple technology stocks
- Evaluate the effectiveness of RNNs for stock price prediction

## Dataset

### Companies Analyzed
- AMZN (Amazon)
- GOOGL (Google/Alphabet)
- IBM (IBM Corporation)
- MSFT (Microsoft)

### Data Features
- **Date**: Trading date from 2006-01-01 to 2018-01-01
- **Open**: Opening price of the stock
- **High**: Highest price during the trading day
- **Low**: Lowest price during the trading day
- **Close**: Closing price (target variable for prediction)
- **Volume**: Number of shares traded
- **Name**: Stock symbol

### Dataset Statistics
- Total Records: 12,076 (after cleaning)
- Time Period: 12 years
- Missing Values: Removed using dropna()

## Technical Implementation

### Data Preprocessing
- Combined multiple CSV files into single dataframe
- Handled missing values by removing incomplete records
- Applied StandardScaler with partial_fit() to prevent data leakage
- Created 20-day sliding windows for time series sequences
- Split data: 80% training, 20% testing

### Model Architectures

**Simple RNN**
- Single SimpleRNN layer with Dense output
- Optimal Configuration: 50 units, 0.2 dropout, 0.001 learning rate
- Performance: MSE: 675,833, MAE: $802.32, R²: -20.05

**Advanced RNN (GRU)**
- Single GRU layer with Dense output
- Optimal Configuration: 64 units, 0.2 dropout, 0.001 learning rate
- Performance: MSE: 457,610, MAE: $622.52, R²: -13.25

### Hyperparameter Tuning
- Simple RNN: 7 configurations tested
- Advanced RNN: 9 configurations tested (LSTM and GRU)
- Selection based on lowest MSE on validation set

## Results

### Model Comparison
| Metric | Simple RNN | Advanced RNN (GRU) | Improvement |
|--------|------------|---------------------|-------------|
| MSE    | 675,833    | 457,610            | +32.3%      |
| MAE    | $802.32    | $622.52            | +22.4%      |
| R²     | -20.05     | -13.25             | +33.9%      |
| MAPE   | 101.0%     | 75.1%              | +25.7%      |

### Key Findings
- GRU outperformed Simple RNN across all metrics
- Both models showed negative R² scores indicating poor predictive performance
- High prediction errors relative to actual stock prices
- 20-day window size effectively captured monthly patterns
- Single-layer architectures performed better than multi-layer ones

## Installation and Usage

### Prerequisites
```
Python 3.8+
TensorFlow 2.x
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Running the Project
1. Install required dependencies
2. Load the stock data CSV files
3. Run the Jupyter notebook for complete analysis
4. View model comparisons and performance metrics

## Limitations

### Model Performance Issues
- Negative R² scores indicate poor predictive accuracy
- High prediction errors limit practical applications
- Models performed worse than simple baseline predictions

### Data Limitations
- Limited to OHLCV data only
- Missing fundamental indicators (P/E ratios, earnings)
- Historical data may not reflect current market dynamics
- No external factors (news, sentiment, economic indicators)

## Conclusion

This project demonstrates both the potential and significant challenges of using RNNs for stock price prediction. While the Advanced RNN (GRU) showed improvement over Simple RNN, both models struggled with accurate predictions due to the inherent complexity and volatility of financial markets. The negative R² scores highlight the difficulty of stock price prediction and the need for more sophisticated approaches that incorporate multiple data sources and market dynamics.

The project serves as a valuable learning experience in applying deep learning to financial time series, showcasing proper data preprocessing, model comparison, and the importance of realistic expectations in financial machine learning applications.
