# Data Cleaner

import pandas as pd

df = pd.read_csv("", skiprows=2) # Input Raw CSV Data From NREL NSRDB Here

df.dropna(subset=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)

df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

df.set_index('Datetime', inplace=True)

df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)

rename_map = {
    'DHI': 'Diffuse Horizontal Irradiance',
    'DNI': 'Direct Normal Irradiance',
    'GHI': 'Global Horizontal Irradiance',
    'Temperature': 'Temperature (°C)',
    'Relative Humidity': 'Humidity (%)',
    'Wind Speed': 'Wind Speed (m/s)',
    'Wind Direction': 'Wind Direction (°)',
    'Pressure': 'Pressure (mbar)',
    'Precipitable Water': 'Precipitable Water (cm)',
    'Aerosol Optical Depth': 'AOD',
    'SSA': 'Single Scattering Albedo',
    'Solar Zenith Angle': 'Solar Zenith Angle (°)'
}
df.rename(columns=rename_map, inplace=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

df.to_csv('data_cleaned.csv')

print(df.head())
