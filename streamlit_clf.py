import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import statsmodels.formula.api as sm
from statsmodels.tsa import tsatools
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set page configuration including a logo and background color
st.set_page_config(page_title="Diamond Price Prediction", 
                   layout="wide",
                   initial_sidebar_state="expanded",
                   )

# Local file
logo_image = "https://github.com/DeepaJames09/DiamondPricePrediction/blob/main/logo.jpeg"
#st.image(logo_image)

col1, col2, col3, col4, col5 = st.columns(5)

with col5:
    st.image(logo_image)

st.markdown("<h1 style='text-align: center;'>Diamond Price Prediction</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# getting user input

cut = col1.selectbox("Enter the diamond cut",["Ideal", "Premium", "Very Good", "Good", "Fair"])

color = col2.selectbox("Enter the diamond color",["D", "E", "F", "G", "H", "I", "J"])

clarity = col3.selectbox("Enter the diamond clarity",["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])

carat = col1.number_input("Enter carat")

depth = col2.number_input("Enter depth")

table = col3.number_input("Enter table")

x = col1.number_input("Enter x")

y = col2.number_input("Enter y")

z = col3.number_input("Enter z")

if st.button('Predict'):


    df_pred = pd.DataFrame([[cut,color,clarity,carat,depth,table,x,y,z]],columns= ['cut','color','clarity','carat','depth','table','x','y','z'])
    
    #Convert column: cut, color, and clarity from categorical to numerical data
    
    df_pred['cut'] = df_pred['cut'].map({'Ideal': 0, 'Premium': 1, 'Very Good': 2, 'Good': 3, 'Fair': 4})
    df_pred['color'] = df_pred['color'].map({'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6})
    df_pred['clarity'] = df_pred['clarity'].map({'IF': 0, 'VVS1': 1, 'VVS2': 2, 'VS1': 3, 'VS2': 4, 'SI1': 5, 'SI2': 6, 'I1': 7})
    df_pred['size'] = df_pred['x']*df_pred['y']*df_pred['z']
    df_pred = np.array(df_pred).reshape(1, -1)
    scaler = joblib.load('scaler.pkl')
    #scaler = StandardScaler()
    df_pred_scaled = scaler.transform(df_pred)

    
    model = joblib.load('clf_model.pkl')
    prediction = model.predict(df_pred_scaled)

    st.write('<p class="big-font"> Predicted Price of Diamond: {} </p>'.format(prediction), unsafe_allow_html=True)


#Forecasting

data_df = pd.read_csv('diamonds_timeseries.csv') 
data_df["date"] = pd.to_datetime(data_df["date"])
data_df = data_df.set_index('date')
diamond_price = data_df['diamond price']
# Exponential Smoothing with seasonality (Holt-Winters) model
model = ExponentialSmoothing(diamond_price, seasonal_periods=12, trend='add', seasonal='add')  # Assuming monthly data
# Fit the model
fit_model = model.fit()
# Forecast for the next 5 years
forecast = fit_model.forecast(steps=5*12)  # Assuming monthly data and forecasting for 5 years
# Generate dates for the forecast
forecast_index = pd.date_range(start=diamond_price.index[-1], periods=5*12+1, freq='M')[1:]
# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(diamond_price.index, diamond_price, label='Actual')
plt.plot(forecast_index, forecast, label='Forecast')
plt.title('Forecast for Diamond Price with Seasonality')
plt.xlabel('Time')
plt.ylabel('Diamond Price')
plt.legend()
plt.show()
plt.savefig("forecast.png")

col1, col2 = st.columns(2)
with col2:
    st.image("forecast.png")












