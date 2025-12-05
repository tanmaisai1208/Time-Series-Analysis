# 📈 Time Series Analysis and Forecasting  

This repository provides a complete pipeline for **time-series analysis**, **pre-processing**, **feature engineering**, **model training**, and **forecasting**.

## ✔️ This Repository Covers  
1. **Introduction to Time Series**  
2. **Data Preparation**  
   - Normalization (0–1)  
   - Stationarity checking (ADF Test)  
3. **Lag Feature Selection**  
4. **Train–Test Splitting**  
5. **Regression, ML, and DL Models**  
   - Model training  
   - Prediction & performance evaluation  

▶️ **To run the project:**  
Run **`Main.py`** — it executes the entire workflow.  
You may also run individual sections by commenting/uncommenting blocks.

Your only required input is your dataset:  
```python
data = your_data


---
 
```markdown
# 1️⃣ What is a Time Series?  

A **time series** is a sequence of observations recorded over time (e.g., stock prices, temperature, heart rate).

### **Time Series Analysis**  
Used to extract **trends**, **seasonality**, **correlations**, and cycles before forecasting.

### **Time Series Forecasting**  
Building models that **predict future values** using historical patterns.

# 2️⃣ Data Pre-processing  

Data preprocessing ensures that the dataset is clean and suitable for modeling.

### 🔹 Missing Values  
Handled using interpolation or pandas functions.

### 🔹 Normalization  
Used when features have different scales. Needed for:  
- KNN  
- Linear Regression  
- Neural Networks  
- Distance-based models  

### 🔹 Standardization  
Transforms data to zero mean & unit variance.

### 🔹 Stationarity Check  
A time series is stationary when:  
- Mean is constant  
- Variance is constant  
- Autocovariance is time-independent  

Check using:  
- Rolling mean/variance  
- **ADF test** (p < 0.05 → stationary)

# 3️⃣ Lag Features (Windowing)  

Lag features represent **previous time steps** used as predictors (lag-1, lag-2, lag-3, etc.).

### **Autocorrelation (ACF)**  
Shows correlation between values and their past values.

### **Partial Autocorrelation (PACF)**  
Shows correlation after removing the effects of earlier lags.  
Helps determine the **optimal lag p** for AR-based models.

# 4️⃣ Train–Test Split  

- **Train set** → used for model training  
- **Validation set** → for tuning hyperparameters  
- **Test set** → for final evaluation  

# 5️⃣ Regression & Forecasting Models  

## 🔷 Linear Models  
- **Linear Regression (LR)**  
- **Least Squares Regression (LS)**  
- **Moving Average (MA)**  
- **Autoregressive (AR)**  
- **ARX Model**  
- **ARIMA (p, d, q)**  

Linear models:  
✔️ Fast & interpretable  
❌ Only linear  
❌ Limited noise handling  
❌ Not suited for multivariate series  
❌ Focus on one-step forecasting

## 🔷 Machine Learning Models  
- **XGBoost Regression**  
- **Linear Regression**  
- **Decision Tree Regression**  
- **Random Forest Regression**

Workflow:  
1. Feed data + engineered features  
2. Train  
3. Test  
4. Predict future values  
  
## 🔷 Deep Learning Models (LSTM)  

LSTM is a powerful recurrent neural network used for sequence prediction.  
It is sensitive to scaling → normalization is required.

### Types  
- **Vanilla LSTM**  
- **Stacked LSTM**  
- **Bidirectional LSTM**  
- **LSTM Autoencoder**  

### Keras Workflow  
1. Define model  
2. Compile  
3. Fit  
4. Evaluate  
5. Predict

# 📦 Install Required Packages  

```bash
pip install numpy
pip install scipy
pip install pandas
pip install seaborn
pip install matplotlib
pip install scikit-learn
pip install keras
  

--- 
```markdown
# 🧠 Main Code Structure  
 
```python
data = sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"]
data, normalize = normalize_data(data, Type_Normalize='MinMaxScaler', Display_Figure='on')
data = test_stationary(data, window=20)
auto_correlation(data, nLags=10)
nLags = 3
train_size = int(len(data) * 0.8)
train_x, train_y = sequences_data(np.array(data[:train_size]), nLags)
test_x, test_y = sequences_data(np.array(data[train_size:]), nLags)


