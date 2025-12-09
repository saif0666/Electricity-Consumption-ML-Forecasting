import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_FILE = 'elec_region_stacked_2005-2023.csv'
TARGET_REGION = 'England'

# Use st.cache_data to load and preprocess the data once
@st.cache_data
def load_and_preprocess_data():
    """Loads data, filters by region, and creates the lag feature."""
    try:
        # Load data, assuming 'Year' column exists
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.error(f"Error: The file '{DATA_FILE}' was not found. Please upload it to your app directory.")
        st.stop()
        
    # Filter for the target region and select relevant columns
    df_model = df[df['Region'] == TARGET_REGION][['Year', 'Domestic_consumption_GWh']].copy()
    
    # Create the Consumption_Lag_1 feature
    df_model['Consumption_Lag_1'] = df_model['Domestic_consumption_GWh'].shift(1)
    
    # Drop the first row which has a NaN for the lag feature
    df_model = df_model.dropna().reset_index(drop=True)
    
    return df_model

# Use st.cache_resource to train the model once
@st.cache_resource
def train_linear_regression_model(df_model):
    """Trains the Linear Regression model on all available data."""
    
    # Define features (X) and target (y)
    # Features used: Year and Consumption_Lag_1
    X = df_model[['Year', 'Consumption_Lag_1']].values 
    y = df_model['Domestic_consumption_GWh'].values
    
    # Initialize scaler and model
    scaler = StandardScaler()
    model = LinearRegression()
    
    # Scale X and train the model
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    return model, scaler, X, y

# --- STREAMLIT APP LAYOUT ---

st.set_page_config(page_title="Electricity Consumption Forecast", layout="centered")

st.title("🔌 UK Domestic Electricity Consumption Forecast (England)")
st.caption(f"Forecasting future consumption using Linear Regression with 'Year' and 'Consumption_Lag_1' features on {TARGET_REGION} data.")

# Load data and train model
df_model = load_and_preprocess_data()
model, scaler, X_train, y_train = train_linear_regression_model(df_model)

st.sidebar.header("Input for Forecast")

# Determine the last year in the dataset
last_year = df_model['Year'].max()
last_consumption = df_model.iloc[-1]['Domestic_consumption_GWh']

# User input for the forecast year (starting from the year after the last known year)
forecast_year = st.sidebar.slider(
    'Year to Forecast',
    min_value=int(last_year + 1),
    max_value=int(last_year + 5),
    value=int(last_year + 1),
    step=1
)

# User input for the most recent consumption value
# This value acts as the 'Consumption_Lag_1' for the forecast year
prev_consumption = st.sidebar.number_input(
    f"Consumption (GWh) for Year {forecast_year - 1}",
    min_value=0.0,
    value=float(last_consumption), # Default to the last known value from data
    step=100.0,
    format="%.2f",
    help=f"The domestic electricity consumption (GWh) recorded in the year preceding the forecast year (i.e., {forecast_year - 1})."
)

# --- PREDICTION LOGIC ---
if st.sidebar.button("Generate Forecast"):
    # Prepare the input for the model
    # The input needs to be scaled using the fitted scaler
    X_predict = np.array([[forecast_year, prev_consumption]])
    
    # The scaler must be fitted on the *training features* (Year and Consumption_Lag_1).
    # To scale a single new data point, we need to apply the existing scaler's mean and std dev.
    # Since the scaler was fit on X_train, we apply the transformation to the new input X_predict
    
    # Find the statistics used to fit the scaler (mean and standard deviation for 'Year' and 'Consumption_Lag_1')
    scaler_mean = scaler.mean_
    scaler_std = scaler.scale_
    
    # Manually scale the new input using the stored statistics
    X_predict_scaled = (X_predict - scaler_mean) / scaler_std
    
    # Make prediction
    prediction_value = model.predict(X_predict_scaled)[0]

    st.subheader(f"Forecast for {forecast_year}")
    
    # Display the result
    st.metric(
        label=f"Domestic Consumption Forecast ({forecast_year})", 
        value=f"{prediction_value:,.2f} GWh"
    )

    # --- PLOT THE RESULTS ---
    st.subheader("Historical Data and Forecast")
    
    # Combine training data and prediction for plotting
    plot_data = pd.DataFrame({
        'Year': df_model['Year'].astype(int).tolist() + [forecast_year],
        'Consumption': df_model['Domestic_consumption_GWh'].tolist() + [prediction_value],
        'Type': ['Historical'] * len(df_model) + ['Forecast']
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Historical Data
    historical = plot_data[plot_data['Type'] == 'Historical']
    ax.plot(historical['Year'], historical['Consumption'], marker='o', label='Historical Consumption', color='cornflowerblue')
    
    # Plot Forecast
    forecast = plot_data[plot_data['Type'] == 'Forecast']
    ax.scatter(forecast['Year'], forecast['Consumption'], marker='X', s=200, color='red', label='Forecast')
    
    # Format the Y-axis to show thousands easily
    def gwh_formatter(x, pos):
        return f'{x:,.0f}'
    ax.yaxis.set_major_formatter(FuncFormatter(gwh_formatter))
    
    # Set X-ticks to include all years, including the forecast year
    all_years = plot_data['Year'].unique()
    ax.set_xticks(all_years)
    ax.tick_params(axis='x', rotation=45)
    
    ax.set_title(f'Domestic Electricity Consumption ({TARGET_REGION})', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Consumption (GWh)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Adjust the input in the sidebar and click 'Generate Forecast' to see the prediction.")