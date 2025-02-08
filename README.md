
# **Stock Price Predictor**  

## **Project Overview**  

Investment firms, hedge funds, and individuals leverage financial models to analyze market behavior and make informed investment decisions. With a vast amount of historical stock prices and company performance data available, machine learning can be a powerful tool in stock price prediction.  

Can we predict stock prices with machine learning? While some theories suggest stock prices are entirely random, top financial firms such as Morgan Stanley and Citigroup invest heavily in quantitative analysts to build predictive models. With the rise of algorithmic trading, nearly **70% of all orders on Wall Street are now placed by software**, signifying the importance of machine learning in finance.  

This project applies **Deep Learning models**, specifically **Long Short-Term Memory (LSTM) Neural Networks**, to predict stock prices. LSTMs are a powerful variant of **Recurrent Neural Networks (RNNs)**, commonly used for sequential data like stock market trends.  

I have used **Keras** and **TensorFlow** to build and compare **16 different models** based on historical stock closing prices and trading volume. The results of these models are analyzed and compared to identify the most accurate predictive approach.  

![Stock Price Predictor](https://github.com/Mahadasghar/stock-predictor/blob/main/data_visualization_lstm_improved.png)  

## **Problem Highlights**  
The challenge of this project is to accurately predict the **future closing price** of a given stock over a specified period. To tackle this, I applied and compared **16 different models**, including:  
- Variations of **LSTM networks**  
- **GRU (Gated Recurrent Unit)** models  
- **ARIMA** for time series forecasting  
- **Hybrid models combining statistical & deep learning techniques**  

The performance of each model was evaluated using **Mean Squared Error (MSE)** and other accuracy metrics.  

### **Achievements:**  
✔ Built and compared **16 different models** for stock price prediction.  
✔ Successfully implemented **Long Short-Term Memory (LSTM) Neural Networks** to forecast future stock prices.  
✔ Achieved a **Mean Squared Error (MSE) of just 0.00093063** in the best-performing model.  
✔ Identified the most effective model by comparing results across all applied techniques.  

### **Key Learnings:**  
✔ How to apply deep learning techniques, specifically **LSTM and GRU** models.  
✔ How to use **Keras & TensorFlow** for stock price prediction.  
✔ How to preprocess and analyze stock market data for machine learning applications.  
✔ How to evaluate and optimize models for improved predictive accuracy.  
✔ How to apply **ensemble and hybrid modeling** techniques to enhance stock price forecasting.  

## **Software and Libraries Used**  
This project utilizes the following technologies:  

- **Programming Language:** Python 3.x  
- **Data Processing & Analysis:** NumPy, Pandas  
- **Machine Learning & Deep Learning:** TensorFlow, Keras, Scikit-Learn  
- **Time Series Forecasting:** Statsmodels, ARIMA  
- **Visualization:** Matplotlib, Seaborn  
- **Development Environment:** Jupyter Notebook  

## **Installation & Setup**  
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/Mahadasghar/Stock-prediction-.git
   cd Stock-prediction-
   ```  
 
2. **Run the Jupyter Notebook or Python script:**  
   ```bash
   jupyter notebook
   ```  
4. **Execute the notebook and compare the performance of different models.**  

## **Project Structure**  
```
📂 Stock-Prediction  
│── 📁 data/           # Stock market dataset & preprocessing scripts  
│── 📁 models/         # Implementations of 16 different models  
│── 📁 notebooks/      # Jupyter notebooks for analysis and comparison  
│── 📁 results/        # Model performance reports & visualizations    
│── README.md         # Project documentation  
```

## **Future Enhancements**  
🚀 Implement real-time stock price prediction using live market data.  
🚀 Optimize models further with advanced hyperparameter tuning techniques.  
🚀 Deploy the best-performing model as a web application for user-friendly predictions.  
