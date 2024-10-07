import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.callbacks import Callback, EarlyStopping, TensorBoard
from datetime import datetime

# Define the base path to the data
data_path = '/work/cmcc/lg07622/work/emulator/data/'

def load_data():
    forcing_files = ['FSDS_timeseriesERA.txt', 'TBOT_timeseriesERA.txt', 'QBOT_timeseriesERA.txt', 
                     'WIND_timeseriesERA.txt', 'PSRF_timeseriesERA.txt', 'PRECTmms_timeseriesERA.txt']
    state_files = ['LEAFC_timeseries.txt', 'LEAFN_timeseries.txt', 'H2OSNO_timeseries.txt', 'H2OSOI_timeseries.txt']

    # Load forcing data (multiple measurements per day with time included)
    forcing_data = {}
    for file in forcing_files:
        print(f"Loading forcing data from: {file}")
        df = pd.read_csv(os.path.join(data_path, file), sep=r'\s+', header=None)

        # Combine the first two columns (Date and Time) into a single Datetime column
        df['Datetime'] = pd.to_datetime(df[0] + ' ' + df[1], errors='coerce')

        # Keep the combined Datetime and the third column (data)
        df = df[['Datetime', 2]]
        df.columns = ['Datetime', file.split('_')[0]]  # Rename columns

        forcing_data[file] = df
        print(f"Forcing data loaded for {file}:")
        print(forcing_data[file].head(), "\n")
    
    # Load state data (1 measurement per day, only dates)
    state_data = {}
    for file in state_files:
        print(f"Loading state data from: {file}")
        df = pd.read_csv(os.path.join(data_path, file), sep=r'\s+', header=None)

        # Only keep the date (the second column is unnecessary)
        df['Date'] = pd.to_datetime(df[0], errors='coerce').dt.date

        # Keep the combined Date and the third column (data)
        df = df[['Date', 2]]
        df.columns = ['Date', file.split('_')[0]]  # Rename columns

        state_data[file] = df
        print(f"State data loaded for {file}:")
        print(state_data[file].head(), "\n")

    return forcing_data, state_data


def prepare_data(forcing_data, state_data):
    print("Preparing data...")

    # Convert Datetime to Date for all forcing dataframes
    for key in forcing_data:
        forcing_data[key]['Date'] = forcing_data[key]['Datetime'].dt.date  # Extract 'Date' from 'Datetime'
        print(f"Converted 'Datetime' to 'Date' for {key}")
    
    # Merge all forcing data into a single dataframe on 'Date'
    print("Merging forcing data...")
    forcing_df = forcing_data['FSDS_timeseriesERA.txt']
    for key in forcing_data:
        if key != 'FSDS_timeseriesERA.txt':
            print(f"Merging {key} into forcing dataframe...")
            forcing_df = pd.merge(
                forcing_df.drop(columns=['Datetime'], errors='ignore'), 
                forcing_data[key].drop(columns=['Datetime'], errors='ignore'), 
                on='Date'
            )
            print(f"Forcing dataframe after merging {key}:")
            print(forcing_df.head(), "\n")

    # Merge all state data on 'Date'
    print("Merging state data...")
    state_df = state_data['LEAFC_timeseries.txt']
    for key in state_data:
        if key != 'LEAFC_timeseries.txt':
            print(f"Merging {key} into state dataframe...")
            state_df = pd.merge(state_df, state_data[key], on='Date')
            print(f"State dataframe after merging {key}:")
            print(state_df.head(), "\n")

    # Combine forcing and state data based on 'Date'
    print("Combining forcing and state data on 'Date'...")
    combined_df = pd.merge(forcing_df, state_df, on='Date')
    print("Combined dataframe (forcing and state data):")
    print(combined_df.head(), "\n")

    # Use shift() to add previous day's state data
    print("Adding previous day's state data using shift()...")
    for col in ['LEAFC', 'LEAFN', 'H2OSNO', 'H2OSOI']:
        combined_df[f'Prev_{col}'] = combined_df[col].shift(1)  # Shift the column by 1 day
    print("Dataframe after adding previous day's state data:")
    print(combined_df[['Date', 'LEAFC', 'Prev_LEAFC', 'LEAFN', 'Prev_LEAFN']].head(), "\n")

    # Drop rows with missing values (due to shift)
    print("Dropping rows with missing values...")
    final_df = combined_df.dropna()
    print("Final dataframe after dropping missing values:")
    print(final_df.head(), "\n")

    return final_df


# Train model and log metrics for TensorBoard
def train_model(final_df):
    # TensorBoard log directory
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Define input features (previous day states + current and previous day forcings)
    features = ['Prev_LEAFC', 'Prev_LEAFN', 'Prev_H2OSNO', 'Prev_H2OSOI',
                'FSDS', 'TBOT', 'QBOT', 'WIND', 'PSRF', 'PRECTmms']  # Current day forcing
    
    # Define target variables (current day states)
    targets = ['LEAFC', 'LEAFN', 'H2OSNO', 'H2OSOI']
    
    X = final_df[features]
    y = final_df[targets]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize Random Forest models for each target variable
    models = {target: RandomForestRegressor(n_estimators=100) for target in targets}
    
    # Train a model for each state variable and log RMSE in TensorBoard
    for i, target in enumerate(targets):
        models[target].fit(X_train, y_train[target])
        
        # Evaluate the model
        y_pred = models[target].predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
        print(f"RMSE for {target}: {rmse}")
        
        # Log RMSE to TensorBoard
        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar(f'RMSE/{target}', rmse, step=i)
    
    return models


# Main execution
if __name__ == "__main__":
    forcing_data, state_data = load_data()
    final_df = prepare_data(forcing_data, state_data)
    models = train_model(final_df)
