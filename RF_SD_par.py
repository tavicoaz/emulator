import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.callbacks import TensorBoard
from joblib import Parallel, delayed
from joblib import dump, load
from datetime import datetime

# Define the base path to the data
data_path = '/work/cmcc/lg07622/work/emulator/data/'
model_dir = '/work/cmcc/lg07622/work/emulator/models/'

def load_data():
    forcing_files = ['FSDS_timeseriesERA.txt', 'TBOT_timeseriesERA.txt', 'QBOT_timeseriesERA.txt', 
                     'WIND_timeseriesERA.txt', 'PSRF_timeseriesERA.txt', 'PRECTmms_timeseriesERA.txt']
    state_files = ['LEAFC_timeseries.txt', 'LEAFN_timeseries.txt', 'H2OSNO_timeseries.txt', 'H2OSOI_timeseries.txt']

    # Load forcing data (multiple measurements per day with time included)
    forcing_data = {}
    for file in forcing_files:
        df = pd.read_csv(os.path.join(data_path, file), sep=r'\s+', header=None)
        df['Datetime'] = pd.to_datetime(df[0] + ' ' + df[1], errors='coerce')
        df = df[['Datetime', 2]]
        df.columns = ['Datetime', file.split('_')[0]]
        forcing_data[file] = df
    
    # Load state data (1 measurement per day, only dates)
    state_data = {}
    for file in state_files:
        df = pd.read_csv(os.path.join(data_path, file), sep=r'\s+', header=None)
        df['Date'] = pd.to_datetime(df[0], errors='coerce').dt.date
        df = df[['Date', 2]]
        df.columns = ['Date', file.split('_')[0]]
        state_data[file] = df

    return forcing_data, state_data

def prepare_data(forcing_data, state_data):
    for key in forcing_data:
        forcing_data[key]['Date'] = forcing_data[key]['Datetime'].dt.date
    
    # Merge all forcing data into a single dataframe on 'Date'
    forcing_df = forcing_data['FSDS_timeseriesERA.txt']
    for key in forcing_data:
        if key != 'FSDS_timeseriesERA.txt':
            forcing_df = pd.merge(
                forcing_df.drop(columns=['Datetime'], errors='ignore'), 
                forcing_data[key].drop(columns=['Datetime'], errors='ignore'), 
                on='Date'
            )

    # Merge all state data on 'Date'
    state_df = state_data['LEAFC_timeseries.txt']
    for key in state_data:
        if key != 'LEAFC_timeseries.txt':
            state_df = pd.merge(state_df, state_data[key], on='Date')

    # Combine forcing and state data based on 'Date'
    combined_df = pd.merge(forcing_df, state_df, on='Date')

    # Shift() to add previous day's state data
    for col in ['LEAFC', 'LEAFN', 'H2OSNO', 'H2OSOI']:
        combined_df[f'Prev_{col}'] = combined_df[col].shift(1)  # Shift by 1 day for previous day states

    # Drop rows with missing values (due to shift)
    final_df = combined_df.dropna()

    return final_df

# Train a single model and save it to disk
def train_single_model(X_train, y_train, X_test, y_test, target, log_dir, index):
    print(f"Starting training for {target}...")
    model = RandomForestRegressor(n_estimators=50, n_jobs=-1)  # Use fewer estimators for faster runs
    model.fit(X_train, y_train[target])

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
    print(f"RMSE for {target}: {rmse}")

    # Log RMSE to TensorBoard
    with tf.summary.create_file_writer(log_dir).as_default():
        tf.summary.scalar(f'RMSE/{target}', rmse, step=index)
    
    # Save the model to a file for later use
    model_filename = os.path.join(model_dir, f'{target}_rf_model.joblib')
    dump(model, model_filename)
    print(f"Saved {target} model to {model_filename}")
    
    return model

# Train model and log metrics for TensorBoard
def train_model(final_df):
    print("Starting model training...")
    # TensorBoard log directory
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define input features (previous day states + current day forcings)
    features = ['Prev_LEAFC', 'Prev_LEAFN', 'Prev_H2OSNO', 'Prev_H2OSOI',
                'FSDS', 'TBOT', 'QBOT', 'WIND', 'PSRF', 'PRECTmms']
    
    # Define target variables (current day states)
    targets = ['LEAFC', 'LEAFN', 'H2OSNO', 'H2OSOI']
    
    X = final_df[features]
    y = final_df[targets]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models for each state variable in parallel using joblib
    models = Parallel(n_jobs=4)(delayed(train_single_model)(X_train, y_train, X_test, y_test, target, log_dir, index)
                                for index, target in enumerate(targets))

    print("Finished model training!")
    return models

# Load a saved model for prediction
def load_model(target):
    model_filename = os.path.join(model_dir, f'{target}_rf_model.joblib')
    if os.path.exists(model_filename):
        print(f"Loading saved model for {target} from {model_filename}...")
        model = load(model_filename)
        return model
    else:
        print(f"Model for {target} not found!")
        return None

# Main execution
if __name__ == "__main__":
    print("Starting the main execution...")
    
    # Load data
    forcing_data, state_data = load_data()
    final_df = prepare_data(forcing_data, state_data)
    
    # Train models
    models = train_model(final_df)

    print("Finished training, starting prediction...")
    
    # Example: Loading a saved model and making a prediction
    target = 'LEAFC'  # Example target
    model = load_model(target)
    
    if model:
        # Ensure X_new has the same feature columns in the same order as during training
        forcing_columns = ['FSDS', 'TBOT', 'QBOT', 'WIND', 'PSRF', 'PRECTmms']
        prev_state_columns = ['Prev_LEAFC', 'Prev_LEAFN', 'Prev_H2OSNO', 'Prev_H2OSOI']

        # Check if model has the attribute 'feature_names_in_' (scikit-learn 1.0+)
        if hasattr(model, 'feature_names_in_'):
            # Ensure the correct order of features by using the model's feature names
            X_new = final_df[model.feature_names_in_].iloc[0:1]  # Select the first row as an example
        else:
            # Default to manually specifying the features if the attribute is unavailable
            X_new = final_df[forcing_columns + prev_state_columns].iloc[0:1]  # Select first row as example

        # Predict the target state using the loaded model
        prediction = model.predict(X_new)
        print(f"Prediction for {target}: {prediction}")

    print("Finished prediction!")
