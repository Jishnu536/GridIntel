print(open("README.md").read())

def detect_anomalies(y_test, y_pred, X_test=None, use_isolation_forest=True, threshold_factor=1):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest
    from datetime import datetime
    import os
    import json
    from uuid import uuid4

    residuals = y_test - y_pred

    # Basic residual-based anomaly detection
    threshold = threshold_factor * np.std(residuals)
    residual_anomalies = abs(residuals) > threshold

    # DataFrame to hold results
    anomaly_df = pd.DataFrame({
        'Date': y_test.index,
        'Actual Output (kWh)': y_test.values,
        'Predicted Output (kWh)': y_pred,
        'Residual': residuals,
        'Residual Anomaly': residual_anomalies
    })

    # Isolation Forest on features if X_test is provided
    if use_isolation_forest and X_test is not None:
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso_preds = iso.fit_predict(X_test)
        anomaly_df['IForest Anomaly'] = iso_preds == -1  # True if anomaly
    else:
        anomaly_df['IForest Anomaly'] = False

    # Combine both anomaly types
    anomaly_df['Anomaly'] = anomaly_df['Residual Anomaly'] | anomaly_df['IForest Anomaly']

    print("\n=== Anomalies Detected ===")
    print(anomaly_df[anomaly_df['Anomaly']])

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(y_test.values, label='Actual', marker='o')
    plt.plot(y_pred, label='Predicted', marker='x')
    plt.scatter(
        anomaly_df[anomaly_df['Anomaly']].index,
        anomaly_df[anomaly_df['Anomaly']]['Actual Output (kWh)'],
        color='red', label='Anomaly', zorder=5
    )
    plt.title("Solar Output Prediction with Detected Anomalies")
    plt.xlabel("Index / Time")
    plt.ylabel("Output (kWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Prepare Zapier Alert Trigger Files
    anomalies_only = anomaly_df[anomaly_df['Anomaly']].copy()

    if not anomalies_only.empty:
        # Add metadata for alerts table
        anomalies_only['Date'] = anomalies_only['Date'].astype(str)
        anomalies_only['Alert ID'] = [str(uuid4()) for _ in range(len(anomalies_only))]
        anomalies_only['Triggered At'] = datetime.now().isoformat()
        anomalies_only['Site'] = "Solar Output"

        # Save anomalies to alerts table (CSV)
        alerts_table_path = "alerts_table.csv"
        if os.path.exists(alerts_table_path):
            existing_df = pd.read_csv(alerts_table_path)
            updated_df = pd.concat([existing_df, anomalies_only], ignore_index=True)
        else:
            updated_df = anomalies_only.copy()
        updated_df.to_csv(alerts_table_path, index=False)
        print(f"Alerts table updated: {alerts_table_path}")

        # Convert datetime index to ISO if needed
        anomalies_only['Date'] = anomalies_only['Date'].astype(str)

        alert_payload = anomalies_only.to_dict(orient="records")

        # Create output directory if it doesn't exist
        output_dir = "zapier_alerts"
        os.makedirs(output_dir, exist_ok=True)

        # Create timestamped file for Zapier to pick up
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        alert_file = os.path.join(output_dir, f"alerts_{timestamp}.json")

        with open(alert_file, "w") as f:
            json.dump(alert_payload, f, indent=2)

        print(f"Zapier alert trigger file created: {alert_file}")

        # Loop through each anomaly and save Zapier-friendly alert
        save_combined_alert_for_zapier(anomalies_only)

        # Trigger Zapier webhook
        webhook_url = "https://hooks.zapier.com/hooks/catch/23632549/u3kptdx/"
        trigger_zapier_webhook(webhook_url, anomalies_only, site_id="Solar Output")

    else:
        print("No anomalies to trigger alert file.")

    return anomaly_df

def save_combined_alert_for_zapier(anomalies_df, site_id="N/A"):
    import json
    from datetime import datetime
    import os

    # Create combined alert content
    body_lines = []
    for _, row in anomalies_df.iterrows():
        line = (
            f"- Date: {row['Date']}\n"
            f"  Output: {row['Actual Output (kWh)']:.2f} kWh\n"
            f"  Predicted: {row['Predicted Output (kWh)']:.2f} kWh\n"
            f"  Residual: {row['Residual']:.2f} kWh"
        )
        body_lines.append(line)

    combined_body = "Multiple anomalies detected:\n\n" + "\n\n".join(body_lines)

    alert = {
        "subject": "Multiple Anomalies Detected",
        "body": combined_body,
        "timestamp": datetime.now().isoformat(),
        "site": site_id
    }

    # Save combined alert to file
    os.makedirs("zapier_emails", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"zapier_emails/combined_alert_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(alert, f, indent=2)

    print(f"Combined Zapier alert saved: {filename}")



def trigger_zapier_webhook(webhook_url, anomalies_df, site_id="Solar Output"):
    import requests
    from datetime import datetime

    if anomalies_df.empty:
        print("No anomalies to send to Zapier.")
        return

    print(f"Sending {len(anomalies_df)} anomalies to Zapier webhook...")

    # Construct email-style body
    body_lines = []
    records = []
    for _, row in anomalies_df.iterrows():
        line = (
            f"Date: {row['Date']}\n"
            f"Output: {row['Actual Output (kWh)']:.2f} kWh\n"
            f"Predicted: {row['Predicted Output (kWh)']:.2f} kWh\n"
            f"Residual: {row['Residual']:.2f} kWh"
        )
        body_lines.append(line)

        # Structure data for Zapier
        records.append({
            "date": row["Date"],
            "actual_output_kwh": round(row["Actual Output (kWh)"], 2),
            "predicted_output_kwh": round(row["Predicted Output (kWh)"], 2),
            "residual": round(row["Residual"], 2)
        })

    combined_message = "\n---\n".join(body_lines)

    payload = {
        "subject": f"Anomaly Alert - {site_id}",
        "body": combined_message,
        "anomaly_count": len(records),
        "timestamp": datetime.now().isoformat(),
        "site": site_id,
        "records": records  # Full structured list of anomalies
    }

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        print("Zap successfully triggered.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to trigger Zap: {e}")

    return payload



def generate_summary(df, daily_ghi, y_test, y_pred, next_date, next_day_prediction, temp_next,
                     humidity_spike_next, irradiance_ratio_daily, daily_humidity_spikes, daily_max_temp,
                     ghi_rolling_next, wind_pct_next, irradiance_ratio_next, X_test, anomaly_df=None):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.dates as mdates

    from datetime import datetime, timedelta
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error

    summary = []

    # 1. General prediction performance
    summary.append(f"Model tested on {len(y_test)} samples.")
    summary.append(f"Average actual output: {np.mean(y_test):.2f} kWh.")
    summary.append(f"Average predicted output: {np.mean(y_pred):.2f} kWh.")
    summary.append(f"Prediction MSE: {mean_squared_error(y_test, y_pred):.2f}.")

    # 2. High output days
    top_days = daily_ghi.sort_values(by='output_kwh', ascending=False).head(3)
    top_str = ', '.join(f"{row['date']}: {row['output_kwh']:.2f} kWh" for _, row in top_days.iterrows())
    summary.append(f"Top 3 energy output days: {top_str}.")

    # 3. Anomaly detection summary (if provided)
    anomaly_df = detect_anomalies(y_test, y_pred, X_test=X_test)
    if anomaly_df is not None:
        anomalies = anomaly_df[anomaly_df['Anomaly']]
        num_anomalies = anomalies.shape[0]
        summary.append(f"Anomalies detected in test data: {num_anomalies} instances.")
        if num_anomalies > 0:
            largest = anomalies.reindex(anomalies['Residual'].abs().sort_values(ascending=False).index).head(1)
            largest_row = largest.iloc[0]
            summary.append(
                f"Largest anomaly: Actual = {largest_row['Actual Output (kWh)']:.2f}, "
                f"Predicted = {largest_row['Predicted Output (kWh)']:.2f}, "
                f"Residual = {largest_row['Residual']:.2f} kWh."
            )
        else:
            summary.append(f"No anomalies detected")
    
    # 4. Next day prediction
    summary.append(f"Predicted output for {next_date.date()}: {next_day_prediction:.2f} kWh.")

    # Collect next-day input features from existing variables
    ghi_7d_sum = df[df['date'] > df['date'].max() - pd.Timedelta(days=7)]['Global Horizontal Irradiance'].sum()
    ghi_rolling_next = ghi_7d_sum / 7
    wind_pct_next = df[df['date'] == df['date'].max()]['wind_speed_pct_change'].mean()
    temp_next = daily_max_temp[daily_max_temp['date'] == df['date'].max()]['daily_max_temp'].values[0]
    humidity_spike_next = daily_humidity_spikes[daily_humidity_spikes['date'] == df['date'].max()]['humidity_spike'].values[0]
    irradiance_ratio_next = irradiance_ratio_daily[irradiance_ratio_daily['date'] == df['date'].max()]['irradiance_ratio'].values[0]
    
    # Create a single-row export DataFrame
    export_df = pd.DataFrame([{
       "Datetime": next_date.date(),
       "Predicted Output (kWh)": next_day_prediction,
       "Maximum Temperature": temp_next,
       "Humidity Spike (>10%)": humidity_spike_next,
       "7 Day Rolling GHI": ghi_rolling_next,
       "% Change in Wind Speed": wind_pct_next,
       "Irradiance/Clearsky Ratio": irradiance_ratio_next,
   }])

    # Save to CSV
    with open("final_output.csv", "w") as f:
        export_df.to_csv(f, index=False)

    # Write summary
    with open("csv_summary.txt", "w") as f:
       f.write("\n".join(summary))

    return summary

def plot_actual(y_test):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_test.values, marker='o', label='Actual (kWh)')
    ax.axhline(y_test.mean(), color='green', linestyle='--', label='Avg Actual')
    ax.set_title("Daily Solar Output Prediction")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Energy Output (kWh)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_prediction_comparison(y_test, y_pred):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_test.values, marker='o', label='Actual (kWh)')
    ax.plot(y_pred, marker='x', label='Predicted (kWh)')
    ax.axhline(y_test.mean(), color='green', linestyle='--', label='Avg Actual')
    ax.axhline(np.mean(y_pred), color='blue', linestyle=':', label='Avg Predicted')
    ax.set_title("Daily Solar Output Prediction")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Energy Output (kWh)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_prediction_scatter(y_test, y_pred):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(y_test, y_pred, alpha=0.6)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax.set_xlabel("Actual Output (kWh)")
    ax.set_ylabel("Predicted Output (kWh)")
    ax.set_title("Prediction Accuracy")
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_anomalies(y_test, y_pred, X_test):
    import matplotlib.pyplot as plt

    anomaly_df = detect_anomalies(y_test, y_pred, X_test=X_test)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_test.values, label='Actual', marker='o')
    ax.plot(y_pred, label='Predicted', marker='x')
    ax.scatter(
        anomaly_df[anomaly_df['Anomaly']].index,
        anomaly_df[anomaly_df['Anomaly']]['Actual Output (kWh)'],
        color='red', label='Anomaly', zorder=5
    )

    ax.set_title("Solar Output Prediction with Detected Anomalies")
    ax.set_xlabel("Index / Time")
    ax.set_ylabel("Output (kWh)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig

def plot_prediction(next_date, next_day_prediction, plot_data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=plot_data, x='date', y='output_kwh', marker='o', label='Actual Output', ax=ax)
    ax.axvline(x=next_date, color='red', linestyle='--', label='Next Day Prediction')
    ax.scatter(
        next_date,
        next_day_prediction,
        color='red',
        s=100,
        zorder=5,
        label=f'Predicted Output: {next_day_prediction:.2f} kWh'
    )

    ax.set_title("Daily Solar Output with Next Day Prediction", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy Output (kWh)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Limit to last 30 days + next day
    start_date = plot_data['date'].max() - pd.Timedelta(days=30)
    ax.set_xlim(start_date, next_date + pd.Timedelta(days=1))
    fig.tight_layout()
    
    return fig



def run_pipeline(csv_path):
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   import matplotlib.dates as mdates


   from sklearn.ensemble import RandomForestRegressor
   from sklearn.model_selection import train_test_split, GridSearchCV
   from sklearn.metrics import mean_squared_error

   # Load and preprocess data
   df = pd.read_csv(csv_path)
   df['Datetime'] = pd.to_datetime(df['Datetime'])
   df['date'] = pd.to_datetime(df['Datetime'].dt.date)

   # Feature Engineering
   df = df.sort_values(by='Datetime')

   # Prediction Accuracy
   def plot_prediction_comparison(y_test, y_pred):
       """
       Plots a line comparison of actual vs predicted values with average lines.
       """
       plt.figure(figsize=(12, 4))
       plt.plot(y_test.values, marker='o', label='Actual (kWh)')
       plt.plot(y_pred, marker='x', label='Predicted (kWh)')
       plt.axhline(np.mean(y_test), color='green', linestyle='--', label='Avg Actual')
       plt.axhline(np.mean(y_pred), color='blue', linestyle=':', label='Avg Predicted')
       plt.title("Daily Solar Output Prediction")
       plt.xlabel("Sample Index")
       plt.ylabel("Energy Output (kWh)")
       plt.legend()
       plt.grid(True)
       plt.tight_layout()
       plt.show()

   # Scatter Plot
   def plot_prediction_scatter(y_test, y_pred):
       """
       Plots a scatter plot comparing actual vs predicted values.
       """
       plt.figure(figsize=(12, 4))
       plt.scatter(y_test, y_pred, alpha=0.6)
       plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
       plt.xlabel("Actual Output (kWh)")
       plt.ylabel("Predicted Output (kWh)")
       plt.title("Prediction Accuracy")
       plt.grid(True)
       plt.tight_layout()
       plt.show()

   def compute_ghi_rolling(df):
       df['GHI_rolling_7d'] = df.groupby('date')['Global Horizontal Irradiance'].transform('sum').rolling(window=7).mean()
       ghi_rolling_daily = df.groupby('date')['GHI_rolling_7d'].mean().reset_index()

       # Plot
       plt.figure(figsize=(12, 4))
       sns.lineplot(x=df['Datetime'], y=df['GHI_rolling_7d'])
       plt.title("7-Day Rolling Avg of GHI")
       plt.xlabel("Datetime")
       plt.ylabel("GHI (W/m²)")
       plt.xticks(rotation=45)
       plt.tight_layout()
       plt.show()

       return df, ghi_rolling_daily

   def compute_wind_speed_pct_change(df):
       wind_speed = df['Wind Speed (m/s)'].replace(0, np.nan)
       df['wind_speed_pct_change'] = wind_speed.pct_change() * 100
       df['wind_speed_pct_change'] = df['wind_speed_pct_change'].replace([np.inf, -np.inf], np.nan).fillna(0)
       wind_pct_daily = df.groupby('date')['wind_speed_pct_change'].mean().reset_index()

       # Plot
       plt.figure(figsize=(12, 4))
       sns.lineplot(x=df['Datetime'], y=df['wind_speed_pct_change'])
       plt.title("Hourly % Change in Wind Speed")
       plt.xlabel("Datetime")
       plt.ylabel("% Change in Wind Speed")
       plt.xticks(rotation=45)
       plt.tight_layout()
       plt.show()

       return df, wind_pct_daily

   def compute_daily_max_temp(df):
       daily_max_temp = df.groupby('date')['Temperature (°C)'].max().reset_index().rename(columns={'Temperature (°C)': 'daily_max_temp'})

       # Plot
       plt.figure(figsize=(12, 4))
       daily_max_temp['date'] = pd.to_datetime(daily_max_temp['date'])
       sns.lineplot(data=daily_max_temp, x='date', y='daily_max_temp')
       plt.title("Daily Maximum Temperature (°C)")
       plt.xlabel("Datetime")
       plt.ylabel("Daily Maximum Temperature (°C)")
       plt.xticks(rotation=45)
       plt.tight_layout()
       plt.show()

       return daily_max_temp

   def compute_humidity_spikes(df):
       df['humidity_spike'] = 0
       for day, group in df.groupby('date'):
           group_sorted = group.sort_values('Datetime')
           humidity_diff = group_sorted['Humidity (%)'].diff().abs()
           spikes = (humidity_diff > 10).astype(int)
           df.loc[group_sorted.index, 'humidity_spike'] = spikes
       daily_humidity_spikes = df.groupby('date')['humidity_spike'].sum().reset_index()

       # Plot
       plt.figure(figsize=(12, 4))
       plt.bar(daily_humidity_spikes['date'], daily_humidity_spikes['humidity_spike'], color='skyblue')
       plt.title("Daily Humidity Spikes (>10%)")
       plt.xlabel("Datetime")
       plt.ylabel("Spike Count")
       plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
       plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
       plt.xticks(rotation=45)
       plt.tight_layout()
       plt.show()

       return df, daily_humidity_spikes

   def compute_irradiance_ratio(df):
       df['irradiance_ratio'] = df['Global Horizontal Irradiance'] / (df['Clearsky GHI'] + 1e-6)
       irradiance_ratio_daily = df.groupby('date')['irradiance_ratio'].mean().reset_index()

       # Plot
       plt.figure(figsize=(12, 4))
       irradiance_ratio_daily['date'] = pd.to_datetime(irradiance_ratio_daily['date'])
       sns.lineplot(data=irradiance_ratio_daily, x='date', y='irradiance_ratio')
       plt.title("Avg Irradiance / Clearsky GHI Ratio")
       plt.xlabel("Datetime")
       plt.ylabel("Avg Irradiance / Clearsky GHI Ratio")
       plt.xticks(rotation=45)
       plt.tight_layout()
       plt.show()

       return df, irradiance_ratio_daily

   # Daily energy output
   panel_area_m2 = 10
   efficiency = 0.18
   daily_ghi = df.groupby('date')['Global Horizontal Irradiance'].sum().reset_index()
   daily_ghi['output_kwh'] = (daily_ghi['Global Horizontal Irradiance'] * panel_area_m2 * efficiency) / 1000
   daily_ghi.drop(columns='Global Horizontal Irradiance', inplace=True)

   # Aggregate daily features
   daily_features = df.groupby('date').agg({
       'Temperature (°C)': 'mean',
       'Humidity (%)': 'mean',
       'Pressure (mbar)': 'mean',
       'Wind Speed (m/s)': 'mean',
       'Clearsky GHI': 'sum',
       'Solar Zenith Angle (°)': 'mean',
       'Cloud Type': 'mean'
   }).reset_index()

   # Feature engineering calculations
   df, ghi_rolling_daily = compute_ghi_rolling(df)
   df, wind_pct_daily = compute_wind_speed_pct_change(df)
   daily_max_temp = compute_daily_max_temp(df)
   df, daily_humidity_spikes = compute_humidity_spikes(df)
   df, irradiance_ratio_daily = compute_irradiance_ratio(df)

   # Merge all engineered features into a new DataFrame
   engineered_features = pd.merge(daily_max_temp, daily_humidity_spikes, on='date', how='inner')
   engineered_features = pd.merge(engineered_features, ghi_rolling_daily, on='date', how='inner')
   engineered_features = pd.merge(engineered_features, wind_pct_daily, on='date', how='inner')
   engineered_features = pd.merge(engineered_features, irradiance_ratio_daily, on='date', how='inner')

   # Merge for ML
   data = pd.merge(daily_ghi, engineered_features, on='date')
   X = data.drop(columns=['date', 'output_kwh'])
   y = data['output_kwh']

   # Train-test split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Model and GridSearchCV
   param_grid = {
       'n_estimators': [100, 200],
       'max_depth': [5, 10, None],
       'min_samples_split': [2, 5],
       'min_samples_leaf': [1, 2]
   }
   rf = RandomForestRegressor(random_state=42)
   print("\n=== Data Integrity Check ===")
   print("Has NaN:", np.isnan(X_train.values).any())
   print("Has inf:", np.isinf(X_train.values).any())
   print("Max value:", np.max(X_train.values))
   print("Min value:", np.min(X_train.values))
   grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
   grid_search.fit(X_train, y_train)

   # Best model and predictions
   best_model = grid_search.best_estimator_
   y_pred = best_model.predict(X_test)

   print("\n=== Best Model Parameters ===")
   print(grid_search.best_params_)
   print("\n=== Model Performance ===")
   print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
   print(f"Avg Actual: {np.mean(y_test):.2f} kWh")
   print(f"Avg Predicted: {np.mean(y_pred):.2f} kWh")

   # Plot Prediction Accuracy
   plot_prediction_comparison(y_test, y_pred)

   # Plot Accuracy Scatter Plot
   plot_prediction_scatter(y_test, y_pred)

   # 1: Plot GHI 7-Day Rolling Avg
   # df, ghi_rolling_daily = compute_ghi_rolling(df)

   # 2: Plot % Change in Wind Speed
   # df, wind_pct_daily = compute_wind_speed_pct_change(df)

   # 3. Plot Daily Max Temperature
   # daily_max_temp = compute_daily_max_temp(df)

   # 4. Plot Daily Humidity Spikes
   # df, daily_humidity_spikes = compute_humidity_spikes(df)

   # 5. Plot Irradiance/Clearsky Ratio
   # df, irradiance_ratio_daily = compute_irradiance_ratio(df)

   # Feature Importance
   def plot_feature_importance(model, feature_names):
       """
       Plots feature importance from a trained model.


       Parameters:
           model: Trained RandomForestRegressor or compatible model with feature_importances_ attribute.
           feature_names: List or Index of feature names used in training.
       """
       importances = model.feature_importances_


       importance_df = pd.DataFrame({
           'Feature': feature_names,
           'Importance': importances
       }).sort_values(by='Importance', ascending=False)


       print("\n=== Feature Impact on Predicted Output (Energy kWh) ===")
       print(importance_df.to_string(index=False))


       plt.figure(figsize=(12, 4))
       sns.barplot(data=importance_df, y='Importance', x='Feature', palette='mako')
       plt.title("Importance of Engineered Features on Solar Output (kWh)")
       plt.ylabel("Importance Score (Percentage Converted to Decimals)")
       plt.xlabel("Engineered Feature")
       plt.grid(True, linestyle='--', alpha=0.6)
       plt.tight_layout()
       plt.show()

       return model, feature_names
   plot_feature_importance(best_model, X.columns)

   # Predict Next Day's Output
   def predict_next_day_output(df, daily_ghi, daily_max_temp, daily_humidity_spikes,
                               irradiance_ratio_daily, best_model):
       """
       Predicts and plots the next day's solar energy output using the trained model and latest data.


       Parameters:
           df (pd.DataFrame): Original dataframe with datetime-indexed weather and solar data.
           daily_ghi (pd.DataFrame): DataFrame containing historical daily GHI energy output.
           daily_max_temp (pd.DataFrame): DataFrame of daily max temperatures.
           daily_humidity_spikes (pd.DataFrame): DataFrame of daily humidity spike counts.
           irradiance_ratio_daily (pd.DataFrame): DataFrame of daily irradiance/clearsky ratios.
           best_model (sklearn model): Trained RandomForestRegressor model.
       """

       # Get the last known date
       last_date = df['date'].max()
       next_date = last_date + pd.Timedelta(days=1)

       # Estimate or use proxy values for engineered features
       ghi_7d_sum = df[df['date'] > last_date - pd.Timedelta(days=7)]['Global Horizontal Irradiance'].sum()
       ghi_rolling_next = ghi_7d_sum / 7

       wind_pct_next = df[df['date'] == last_date]['wind_speed_pct_change'].mean()
       temp_next = daily_max_temp[daily_max_temp['date'] == last_date]['daily_max_temp'].values[0]
       humidity_spike_next = daily_humidity_spikes[daily_humidity_spikes['date'] == last_date]['humidity_spike'].values[0]
       irradiance_ratio_next = irradiance_ratio_daily[irradiance_ratio_daily['date'] == last_date]['irradiance_ratio'].values[0]

       # Build feature row
       next_day_features = pd.DataFrame([{
           'daily_max_temp': temp_next,
           'humidity_spike': humidity_spike_next,
           'GHI_rolling_7d': ghi_rolling_next,
           'wind_speed_pct_change': wind_pct_next,
           'irradiance_ratio': irradiance_ratio_next
       }])

       # Predict
       next_day_prediction = best_model.predict(next_day_features)[0]

       print("\n=== Predicted Output for Next Day ({}) ===".format(next_date.date()))
       print(f"Predicted Solar Energy Output: {next_day_prediction:.2f} kWh")

       # Append and plot
       plot_data = daily_ghi.copy()
       plot_data['date'] = pd.to_datetime(plot_data['date'])
       next_row = pd.DataFrame({
           'date': [next_date],
           'output_kwh': [next_day_prediction]
       })
       plot_data = pd.concat([plot_data, next_row], ignore_index=True)

       plt.figure(figsize=(12, 4))
       sns.lineplot(data=plot_data, x='date', y='output_kwh', marker='o', label='Actual Output')
       plt.axvline(x=next_date, color='red', linestyle='--', label='Next Day Prediction')
       plt.scatter(
           next_date,
           next_day_prediction,
           color='red',
           s=100,
           zorder=5,
           label=f'Predicted Output: {next_day_prediction:.2f} kWh'
       )
       plt.title("Daily Solar Output with Next Day Prediction", fontsize=14)
       plt.xlabel("Date")
       plt.ylabel("Energy Output (kWh)")
       plt.xticks(rotation=45)
       plt.legend()
       plt.grid(True, linestyle='--', alpha=0.6)

       start_date = plot_data['date'].max() - pd.Timedelta(days=30)
       plt.xlim(start_date, next_date + pd.Timedelta(days=1))

       plt.tight_layout()
       plt.show()

       return next_day_prediction, next_date, plot_data, temp_next, humidity_spike_next, ghi_rolling_next, wind_pct_next, irradiance_ratio_next

   next_day_prediction, next_date, plot_data, temp_next, humidity_spike_next, ghi_rolling_next, wind_pct_next, irradiance_ratio_next = predict_next_day_output(
       df=df,
       daily_ghi=daily_ghi,
       daily_max_temp=daily_max_temp,
       daily_humidity_spikes=daily_humidity_spikes,
       irradiance_ratio_daily=irradiance_ratio_daily,
       best_model=best_model
   )

   anomaly_df = detect_anomalies(y_test, y_pred, X_test=None, use_isolation_forest=True, threshold_factor=1)

   generate_summary(df, daily_ghi, y_test, y_pred, next_date, next_day_prediction, temp_next,
                     humidity_spike_next, irradiance_ratio_daily, daily_humidity_spikes, daily_max_temp,
                     ghi_rolling_next, wind_pct_next, irradiance_ratio_next, X_test, anomaly_df=None)
   
   return X_train, X_test, y_train, y_test, y_pred, daily_ghi, next_date, next_day_prediction, plot_data
   


run_pipeline("/Users/jishnu/Downloads/NREL NSRDB Datasets 2018-23/Cleaned Data/2023_data_cleaned.csv")