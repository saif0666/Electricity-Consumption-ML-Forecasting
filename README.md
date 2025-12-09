# ⚡ Electricity Consumption ML Forecasting

This project focuses on forecasting electricity consumption using Machine Learning techniques.  
The goal is to help manage power usage more efficiently and enable better decision-making for energy providers and consumers.  
This work was developed as part of an undergraduate research project.

---

## 🎯 Objective

- Analyze historical electricity consumption patterns  
- Build and evaluate ML models to predict future consumption  
- Visualize the forecast results to communicate insights effectively  

---

## 📂 Dataset

- Source: Publicly available electricity consumption dataset  
- Key Features:
  - Timestamp-based consumption records
  - Features such as temperature, humidity, seasons, etc. *(if applicable)*  
- Data preprocessing involved:
  - Handling missing values
  - Feature scaling
  - Outlier removal using IQR
  - Converting date/time into useful features (hour, day, month, etc.)

---

## 🧠 Machine Learning Models Used

| Model | Purpose | Result |
|-------|---------|--------|
| Linear Regression | Baseline prediction | Good linear trend mapping |
| Random Forest Regressor | Higher prediction accuracy | Captures non-linear behavior |
| Other tested models *(optional)* | — | — |

Performance evaluation metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

---

## 📈 Visualization

To make the results understandable to everyone, the model predictions were visualized using:
- Actual vs Predicted graph
- Time-series forecasting plot
- Error distribution chart

> The visual representation helps clearly observe how well the model predicts future electricity usage.

---

## 🔧 Tools & Technologies

- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- (Optional) Prophet, TensorFlow, or other libraries

---

## 📊 Output Example
<img width="1226" height="919" alt="electricity-consumption-ml-forecasting-sw8homzagr9ipdkmx6lpbr streamlit app_ (1)" src="https://github.com/user-attachments/assets/a3287f87-5e87-49ed-8ec4-f15af141d6e3" />


