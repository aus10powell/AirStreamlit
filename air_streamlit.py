import streamlit as st
import serial
import time
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import deque
import threading
from datetime import datetime, timedelta
import os

# Configure the serial port
ser = serial.Serial('/dev/ttyUSB0')  # You might need to change this to the correct port
ser.baudrate = 9600

# File to store historical data
HISTORY_FILE = 'air_quality_history.parquet'

# Create deques to store the data
max_points = 3600  # Store up to 1 hour of data (1 reading per second)
times = deque(maxlen=max_points)
pm25_values = deque(maxlen=max_points)
pm10_values = deque(maxlen=max_points)

# Create a buffer for 15 seconds of data
buffer_times = []
buffer_pm25 = []
buffer_pm10 = []

# Flag to control the sensor reading thread
stop_thread = threading.Event()

def append_to_parquet(new_data):
    """
    Append new data to the Parquet file.
    """
    try:
        # Convert time column to string
        new_data['time'] = new_data['time'].astype(str)
        
        # Read existing data
        if os.path.exists(HISTORY_FILE):
            existing_data = pd.read_parquet(HISTORY_FILE)
            # Convert existing time column to string if it's not already
            existing_data['time'] = existing_data['time'].astype(str)
            # Append new data
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            updated_data = new_data

        # Write updated data to Parquet file
        updated_data.to_parquet(HISTORY_FILE, engine='pyarrow', index=False)
    except Exception as e:
        print(f"Error appending to Parquet file: {e}")

def read_pm_values():
    """
    Read PM2.5 and PM10 values from the serial port

    Returns:
    - pm25: PM2.5 value
    - pm10: PM10 value
    """
    data = []
    for _ in range(10):
        datum = ser.read()
        data.append(datum)
    pm25 = int.from_bytes(b''.join(data[2:4]), byteorder='little') / 10
    pm10 = int.from_bytes(b''.join(data[4:6]), byteorder='little') / 10
    return pm25, pm10

def sensor_thread():
    while not stop_thread.is_set():
        pm25, pm10 = read_pm_values()
        current_time = datetime.now()
        
        # Add data to the buffer
        buffer_times.append(current_time)
        buffer_pm25.append(pm25)
        buffer_pm10.append(pm10)
        
        time.sleep(1)

def calculate_median(times_deque, values_deque, time_range):
    if not values_deque:
        return 0
    current_time = datetime.now()
    
    # Create copies of the deques to avoid modification during iteration
    times_copy = list(times_deque)
    values_copy = list(values_deque)
    
    filtered_values = [v for t, v in zip(times_copy, values_copy) if current_time - t <= time_range]
    return np.nanmedian(filtered_values)

def trim_outliers(data, column):
    """
    Remove outliers from a DataFrame based on the IQR method
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    trimmed_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    has_outliers = len(data) != len(trimmed_data)
    return trimmed_data, has_outliers

def load_historical_data():
    """
    Load historical data from Parquet file.
    """
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_parquet(HISTORY_FILE)
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S.%f')
            return df
        except FileNotFoundError:
            return pd.DataFrame(columns=['time', 'PM2.5', 'PM10'])
    else:
        return pd.DataFrame(columns=['time', 'PM2.5', 'PM10'])

def calculate_average(times_deque, values_deque, time_range):
    if not values_deque:
        return 0

    current_time = datetime.now()

    # Create copies of the deques to avoid modification during iteration
    times_copy = list(times_deque)
    values_copy = list(values_deque)

    # Filter values based on time_range
    filtered_values = [v for t, v in zip(times_copy, values_copy) if current_time - t <= time_range]
    
    if not filtered_values:
        return 0

    # Calculate the 1st and 99th percentiles
    lower_bound = np.percentile(filtered_values, 1)
    upper_bound = np.percentile(filtered_values, 99)

    # Filter the values within the 1st and 99th percentile
    trimmed_values = [v for v in filtered_values if lower_bound <= v <= upper_bound]

    return sum(trimmed_values) / len(trimmed_values) if trimmed_values else 0

# Streamlit app
st.title('Indoor Air Quality Monitor')

# Load historical data
historical_data = load_historical_data()

# Create placeholders for chart and averages
chart_placeholder = st.empty()
averages_placeholder = st.empty()

# Start the sensor thread automatically
thread = threading.Thread(target=sensor_thread)
thread.start()

# Main loop
try:
    while True:
        # Wait for 15 seconds to accumulate data
        time.sleep(15)
        
        # Process the buffered data
        if buffer_times:
            # Create a DataFrame from the buffer
            new_data = pd.DataFrame({
                'time': buffer_times,
                'PM2.5': buffer_pm25,
                'PM10': buffer_pm10
            })
            
            # Append to Parquet file
            append_to_parquet(new_data)
            
            # Clear the buffer
            buffer_times.clear()
            buffer_pm25.clear()
            buffer_pm10.clear()
            
            # Reload historical data
            historical_data = load_historical_data()
            
            # Update the deques with the latest hour of data
            last_hour = datetime.now() - timedelta(hours=1)
            recent_data = historical_data[historical_data['time'] > last_hour]
            times.clear()
            pm25_values.clear()
            pm10_values.clear()
            times.extend(recent_data['time'])
            pm25_values.extend(recent_data['PM2.5'])
            pm10_values.extend(recent_data['PM10'])

        # Calculate averages
        hour_ago = timedelta(hours=1)
        day_ago = timedelta(hours=24)
        pm25_median_1h = calculate_median(times, pm25_values, hour_ago)
        pm10_median_1h = calculate_median(times, pm10_values, hour_ago)
        pm25_avg_24h = calculate_average(times, pm25_values, day_ago)
        pm10_avg_24h = calculate_average(times, pm10_values, day_ago)

        # Update averages display at the top
        averages_text = f"""
        Past hour median - PM2.5: {pm25_median_1h:.2f} µg/m³, PM10: {pm10_median_1h:.2f} µg/m³ \n
        Past 24 hours average - PM2.5: {pm25_avg_24h:.2f} µg/m³, PM10: {pm10_avg_24h:.2f} µg/m³
        """
        averages_placeholder.write(averages_text)

        # Create a DataFrame from the deques
        df = pd.DataFrame({
            'time': list(times),
            'PM2.5': list(pm25_values),
            'PM10': list(pm10_values)
        })


        # Create the combined plot with three subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.1,
                            subplot_titles=("Air Quality Measurements", 
                                            "Hourly Distribution (Last 24 Hours)",
                                            "Daily Distribution (Last 7 Days)"),
                            row_heights=[0.5, 0.25, 0.25])

        # Trim outliers for the main chart
        trimmed_df_pm25, has_outliers_pm25 = trim_outliers(df, 'PM2.5')
        trimmed_df, has_outliers_pm10 = trim_outliers(trimmed_df_pm25, 'PM10')

        # Add time series traces
        fig.add_trace(go.Scatter(
            x=trimmed_df['time'], y=trimmed_df['PM2.5'], mode='markers', name='PM2.5',
            marker=dict(color='blue', size=6)), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trimmed_df['time'], y=trimmed_df['PM10'], mode='markers', name='PM10',
            marker=dict(color='lightblue', size=6)), row=1, col=1)

        # Add shaded areas and limit lines
        fig.add_trace(go.Scatter(
            x=trimmed_df['time'].tolist() + trimmed_df['time'].tolist()[::-1],
            y=[6] * len(trimmed_df) + [0] * len(trimmed_df),
            fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='PM2.5 Healthy Zone', showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trimmed_df['time'].tolist() + trimmed_df['time'].tolist()[::-1],
            y=[15] * len(trimmed_df) + [0] * len(trimmed_df),
            fill='tozeroy', fillcolor='rgba(173, 216, 230, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='PM10 Healthy Zone', showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trimmed_df['time'], y=[6] * len(trimmed_df), mode='lines',
            name='PM2.5 Healthy Upper Limit',
            line=dict(color='rgba(0, 0, 255, 0.5)', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trimmed_df['time'], y=[15] * len(trimmed_df), mode='lines',
            name='PM10 Healthy Upper Limit',
            line=dict(color='rgba(173, 216, 230, 0.5)', dash='dash')), row=1, col=1)

        # Add annotation for extreme values in the main chart
        if has_outliers_pm25 or has_outliers_pm10:
            fig.add_annotation(
                x=1, y=1, 
                text="* Extreme values present", 
                showarrow=False, 
                xref="x domain", yref="y domain",
                font=dict(size=10, color="red"),
                row=1, col=1
            )

        # Prepare data for hourly box plots
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        hourly_data = historical_data[(historical_data['time'] > start_time) & (historical_data['time'] <= end_time)]
        hourly_data['hour'] = hourly_data['time'].dt.floor('H')

        # Create a list of hour labels for x-axis
        hour_labels = [(start_time + timedelta(hours=i)).strftime('%Y-%m-%d %H:00') for i in range(24)]

        # Trim outliers and add hourly box plots
        for i, (pollutant, color) in enumerate([('PM2.5', 'blue'), ('PM10', 'lightblue')]):
            trimmed_data, has_outliers = trim_outliers(hourly_data, pollutant)
            
            fig.add_trace(go.Box(
                y=trimmed_data[pollutant],
                x=trimmed_data['hour'],
                name=pollutant,
                marker_color=color,
                showlegend=False,
                offsetgroup=i  # This will offset the boxes side by side
            ), row=2, col=1)
            
            if has_outliers:
                fig.add_annotation(
                    x=1, y=1, 
                    text=f"* Extreme values present in {pollutant}", 
                    showarrow=False, 
                    xref="x2 domain", yref="y2 domain",
                    font=dict(size=10, color="red"),
                    row=2, col=1
                )

        # Prepare data for weekly box plots
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        weekly_data = historical_data[(historical_data['time'] > start_time) & (historical_data['time'] <= end_time)]
        weekly_data['date'] = weekly_data['time'].dt.date

        # Create a list of date labels for x-axis
        date_labels = [(start_time + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]

        # Trim outliers and add weekly box plots
        for i, (pollutant, color) in enumerate([('PM2.5', 'blue'), ('PM10', 'lightblue')]):
            trimmed_data, has_outliers = trim_outliers(weekly_data, pollutant)
            
            fig.add_trace(go.Box(
                y=trimmed_data[pollutant],
                x=trimmed_data['date'],
                name=pollutant,
                marker_color=color,
                showlegend=False,
                offsetgroup=i  # This will offset the boxes side by side
            ), row=3, col=1)
            
            if has_outliers:
                fig.add_annotation(
                    x=1, y=1, 
                    text=f"* Extreme values present in {pollutant}", 
                    showarrow=False, 
                    xref="x3 domain", yref="y3 domain",
                    font=dict(size=10, color="red"),
                    row=3, col=1
                )

        # Update layout
        fig.update_layout(
            height=1000,  # Increased height to accommodate the new subplot
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255, 255, 255, 0.5)"  # Semi-transparent background
            ),
            yaxis_title="Concentration (µg/m³)",
            yaxis2_title="Concentration (µg/m³)",
            yaxis3_title="Concentration (µg/m³)",
            boxmode='group'  # This ensures the boxes are grouped side by side
        )

        # Update x-axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(
            title_text="Time (Last 24 Hours)", 
            tickmode='array',
            tickvals=hour_labels,
            ticktext=[datetime.strptime(label, '%Y-%m-%d %H:00').strftime('%I %p') for label in hour_labels],
            tickangle=45,
            row=2, col=1
        )
        fig.update_xaxes(
            title_text="Date (Last 7 Days)", 
            tickmode='array',
            tickvals=date_labels,
            ticktext=[datetime.strptime(label, '%Y-%m-%d').strftime('%a, %b %d') for label in date_labels],
            tickangle=45,
            row=3, col=1
        )
        
        # Update y-axis range for the main plot based on trimmed data
        y_max = max(max(trimmed_df['PM2.5'].max(), trimmed_df['PM10'].max()), 15) * 1.1
        fig.update_yaxes(range=[0, y_max], row=1, col=1)

        # Update the chart
        chart_placeholder.plotly_chart(fig, use_container_width=True)

except KeyboardInterrupt:
    print("Measurement stopped by user")
finally:
    stop_thread.set()
    ser.close()