# Solar Power Forecasting Pipeline

## What This Workflow Does

This pipeline forecasts **next-day solar energy output (in kWh)** using historical weather and solar irradiance data. It performs:

- Data loading and preprocessing
- Feature engineering (weather-based predictors)
- Model training using Random Forest Regression
- Evaluation and visualizations of predictions
- **Anomaly detection using residual analysis and Isolation Forest**
- **Generation of alert tables for anomalies**
- **Integration with Zapier for workflow automation**
- **Interactive frontend via Streamlit**

---

## Model and Metrics

**Model Used**:  
- `RandomForestRegressor` from `sklearn.ensemble`

**Engineered Features**:
- 7-day rolling average of GHI (Global Horizontal Irradiance)
- % change in wind speed
- Daily maximum temperature
- Humidity spike count (>10% change)
- Irradiance-to-clearsky irradiance ratio

**Evaluation Metric**:
- Mean Squared Error (MSE) on test predictions

---

## Outputs for the User

After running the pipeline, you’ll receive:

- Line chart comparing predicted vs. actual solar output  
- Scatter plot to visualize prediction accuracy  
- Summary Text Report (`weekly_summary.txt`)
- CSV File (`final_output_with_summary.csv`) with:
  - Next day’s forecasted kWh output
  - Engineered feature values for that day
  - Full summary as a column  
- **Anomalies chart showing outlier detections**
- **Alerts Table (`alerts_table.csv`) for any detected anomalies**
- **Zapier JSON trigger file for external workflows**
- **Streamlit dashboard for interactive exploration**

---

## Anomaly Detection and Alerts

The pipeline now includes robust anomaly detection by combining:

- Residual-based anomaly flagging using prediction error thresholds
- Isolation Forest to detect feature-space outliers

Detected anomalies are:

- Plotted for easy visualization
- Saved to `alerts_table.csv`
- Pushed to Zapier via webhook with alert details

Each alert includes:
- Type of anomaly
- Severity
- Timestamp
- Actual vs. predicted values
- Residuals and outlier flags

---

## Streamlit Dashboard

The project includes a live frontend powered by **Streamlit**:

### Pages:
- **CSV Input** – Upload and process new datasets
- **Home** – View prediction charts and tomorrow’s forecast  
- **Anomalies** – Inspect anomaly plots and corresponding alerts table  

### Features:
- Authenticated access
- Dynamic charts
- Real-time Alerts Table
- Text summary display
- Zapier webhook integration for alerts

---

## How to Run This Project

> Make sure you have Python 3.8+ and `pip` installed.

### 1. Clone This Repository

git clone https://github.com/Jishnu536/GridIntel
cd GridIntel

### 2. Install Dependencies

pip install -r requirements

### 3. Add Your Data

Place your cleaned CSV dataset in the project folder, and/or upload it directly via the Streamlit app.

### 4. Launch the. Dashboard

python3 -m streamlit run app.py

### 5. Use the Interface

- Upload your CSV dataset
- View prediction plots and next-day forecast
- Inspect anomaly detection results
- Download alerts and summary reports
