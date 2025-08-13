# 📈 Stock Price Predictor

Welcome to the Stock Price Predictor project! This repository contains a comprehensive Jupyter Notebook that demonstrates how to use various machine learning and deep learning models to forecast stock prices based on historical data.

## 🚀 Features

- **Data Collection:** Automatically downloads historical stock data using [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`.
- **Preprocessing:** Cleans, normalizes, and prepares data for modeling.
- **Visualization:** Plots raw and processed data for quick insights.
- **Modeling:** Implements and compares 12+ models, including:
  - Linear Regression
  - LSTM (Basic & Improved)
  - CNN
  - GRU
  - RNN
  - XGBoost
  - SVR
  - Transformer
  - Random Forest
  - LightGBM
  - Gradient Boosting
  - MLP
  - ElasticNet, Ridge, Lasso
- **Performance Comparison:** Summarizes and visualizes model performance (MSE, RMSE) to help select the best predictor.
- **Robustness Testing:** Evaluates models on unseen data.

## 📝 How to Use

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/stock-predictor.git
   cd stock-predictor
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the notebook:**
   - Open `Stock_Price_Predictor.ipynb` in [Jupyter Notebook](https://jupyter.org/) or [VS Code](https://code.visualstudio.com/).
   - Execute cells step by step to follow the workflow.

## 📊 Results

- The notebook outputs a comparison table and visualizations for all models.
- The best performing model is highlighted based on test MSE/RMSE.

## 📂 Project Structure

```
stock-predictor/
├── Stock_Price_Predictor.ipynb
├── preprocess_data.py
├── lstm.py
├── LinearRegressionModel.py
├── stock_data.py
├── visualize.py
├── data_visualization_*.png
├── google.csv / google_preprocessed.csv / googl.csv
├── LICENSE
└── requirements.txt
```

## 🏆 Model Selection

At the end of the notebook, you’ll find a summary of all models and a clear indication of which model gave the best prediction for your dataset.

## 📚 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

> **Made with ❤️ for learning and
