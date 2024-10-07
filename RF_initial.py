import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define the base path to the data
data_path = '/work/cmcc/lg07622/work/emulator/data/'

# Load atmospheric forcing and model state data
def load_data():
    forcing_files = ['FSDS_timeseriesERA.txt', 'TBOT_timeseriesERA.txt', 'QBOT_timeseriesERA.txt', 
                     'WIND_timeseriesERA.txt', 'PSRF_timeseriesERA.txt', 'PRECTmms_timeseriesERA.txt']
    state_files = ['LEAFC_timeseries.txt', 'LEAFN_timeseries.txt', 'H2OSNO_timeseries.txt', 'H2OSOI_timeseries.txt']
    
    # Load forcing data (4 values per day) and parse datetime from the first column
    forcing_data = {}
    for file in forcing_files:
        df = pd.read_csv(os.path.join(data_path, file), delim_whitespace=True, header=None, names=['Datetime', file.split('_')[0]])
        df['Datetime'] = pd.to_datetime(df['Datetime'])  # Combine Date and Time into a single column
        forcing_data[file] = df
    
    # Load model state data (1 value per day)
    state_data = {}
    for file in state_files:
        df = pd.read_csv(os.path.join(data_path, file), delim_whitespace=True, header=None, names=['Date', file.split('_')[0]])
        df['Date'] = pd.to_datetime(df['Date'])  # Parse Date column
        state_data[file] = df
    
    return forcing_data, state_data

# Prepare the data by combining previous day's state and forcing data
def prepare_data(forcing_data, state_data):
    # Merge forcing data (which contains 'Datetime')
    forcing_df = forcing_data['FSDS_timeseriesERA.txt']
    for key, value in forcing_data.items():
        if key != 'FSDS_timeseriesERA.txt':
            forcing_df = forcing_df.merge(value, on='Datetime')
    
    # Merge state data (which contains 'Date')
    state_df = state_data['LEAFC_timeseries.txt']
    for key, value in state_data.items():
        if key != 'LEAFC_timeseries.txt':
            state_df = state_df.merge(value, on='Date')
    
    # Combine forcing and state data based on 'Date'
    forcing_df['Date'] = forcing_df['Datetime'].dt.date  # Extract Date from Datetime for merging
    combined_df = pd.merge(forcing_df.drop(columns=['Datetime']), state_df, on='Date')
    
    # Prepare input: Use forcing data from previous and current day, and state from the previous day
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df['Previous_Date'] = combined_df['Date'] - pd.Timedelta(days=1)
    
    prev_state_df = combined_df[['Date', 'LEAFC', 'LEAFN', 'H2OSNO', 'H2OSOI']].copy()
    prev_state_df.columns = ['Previous_Date', 'Prev_LEAFC', 'Prev_LEAFN', 'Prev_H2OSNO', 'Prev_H2OSOI']
    
    final_df = pd.merge(combined_df, prev_state_df, on='Previous_Date')
    
    # Drop rows with missing previous day values (if at start of time series)
    final_df.dropna(inplace=True)
    
    return final_df

# Train model
def train_model(final_df):
    # Define input features (previous day states + current and previous day forcings)
    features = ['Prev_LEAFC', 'Prev_LEAFN', 'Prev_H2OSNO', 'Prev_H2OSOI',
                'FSDS_x', 'TBOT_x', 'QBOT_x', 'WIND_x', 'PSRF_x', 'PRECTmms_x',  # Current day forcing
                'FSDS_y', 'TBOT_y', 'QBOT_y', 'WIND_y', 'PSRF_y', 'PRECTmms_y']  # Previous day forcing
    
    # Define target variables (current day states)
    targets = ['LEAFC', 'LEAFN', 'H2OSNO', 'H2OSOI']
    
    X = final_df[features]
    y = final_df[targets]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize Random Forest models for each target variable
    models = {target: RandomForestRegressor(n_estimators=100) for target in targets}
    
    # Train a model for each state variable
    for target in targets:
        models[target].fit(X_train, y_train[target])
    
    # Evaluate the models
    for target in targets:
        y_pred = models[target].predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
        print(f"RMSE for {target}: {rmse}")
    
    return models

# Main execution
if __name__ == "__main__":
    forcing_data, state_data = load_data()
    final_df = prepare_data(forcing_data, state_data)
    models = train_model(final_df)
