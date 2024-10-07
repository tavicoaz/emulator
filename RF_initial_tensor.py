import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import logging
import psutil  # To monitor memory usage

# Setup logging
logging.basicConfig(filename='/work/cmcc/lg07622/work/emulator/emulator_log.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Function to monitor memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f'Memory Usage: RSS = {mem_info.rss / 1024**3:.2f} GB')

# Define the base path to the data
data_path = '/work/cmcc/lg07622/work/emulator/data/'

# Load atmospheric forcing and model state data
def load_data():
    logging.info('Loading data...')
    forcing_files = ['FSDS_timeseriesERA.txt', 'TBOT_timeseriesERA.txt', 'QBOT_timeseriesERA.txt', 
                     'WIND_timeseriesERA.txt', 'PSRF_timeseriesERA.txt', 'PRECTmms_timeseriesERA.txt']
    state_files = ['LEAFC_timeseries.txt', 'LEAFN_timeseries.txt', 'H2OSNO_timeseries.txt', 'H2OSOI_timeseries.txt']
    
    forcing_data = {}
    for file in forcing_files:
        df = pd.read_csv(os.path.join(data_path, file), delim_whitespace=True, header=None, names=['Datetime', file.split('_')[0]])
        df['Datetime'] = pd.to_datetime(df['Datetime'])  # Combine Date and Time into a single column
        forcing_data[file] = df
        logging.info(f'Loaded {file} with shape {df.shape}')
    
    state_data = {}
    for file in state_files:
        df = pd.read_csv(os.path.join(data_path, file), delim_whitespace=True, header=None, names=['Date', file.split('_')[0]])
        df['Date'] = pd.to_datetime(df['Date'])  # Parse Date column
        state_data[file] = df
        logging.info(f'Loaded {file} with shape {df.shape}')
    
    log_memory_usage()
    return forcing_data, state_data

# Prepare the data by combining previous day's state and forcing data
def prepare_data(forcing_data, state_data):
    logging.info('Preparing data...')
    
    forcing_df = forcing_data['FSDS_timeseriesERA.txt']
    for key, value in forcing_data.items():
        if key != 'FSDS_timeseriesERA.txt':
            forcing_df = forcing_df.merge(value, on='Datetime')
            logging.info(f'Merged {key} into forcing_df, new shape {forcing_df.shape}')
    
    state_df = state_data['LEAFC_timeseries.txt']
    for key, value in state_data.items():
        if key != 'LEAFC_timeseries.txt':
            state_df = state_df.merge(value, on='Date')
            logging.info(f'Merged {key} into state_df, new shape {state_df.shape}')
    
    forcing_df['Date'] = forcing_df['Datetime'].dt.date  # Extract Date from Datetime for merging
    combined_df = pd.merge(forcing_df.drop(columns=['Datetime']), state_df, on='Date')
    logging.info(f'Combined forcing and state data, new shape {combined_df.shape}')
    
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df['Previous_Date'] = combined_df['Date'] - pd.Timedelta(days=1)
    
    prev_state_df = combined_df[['Date', 'LEAFC', 'LEAFN', 'H2OSNO', 'H2OSOI']].copy()
    prev_state_df.columns = ['Previous_Date', 'Prev_LEAFC', 'Prev_LEAFN', 'Prev_H2OSNO', 'Prev_H2OSOI']
    
    final_df = pd.merge(combined_df, prev_state_df, on='Previous_Date')
    final_df.dropna(inplace=True)
    
    logging.info(f'Final data prepared, shape {final_df.shape}')
    log_memory_usage()
    return final_df

# Normalize the features
def normalize_data(X_train, X_test):
    logging.info('Normalizing data...')
    scaler = StandardScaler()  # Standardize to zero mean and unit variance
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info('Data normalization completed.')
    log_memory_usage()
    return X_train_scaled, X_test_scaled, scaler

# Function to train a single model (to be run in parallel)
def train_single_model(X_train, y_train, X_test, y_test, target):
    logging.info(f'Training model for {target}...')
    model = RandomForestRegressor(n_estimators=50)  # Reduced number of estimators for memory optimization
    model.fit(X_train, y_train[target])
    logging.info(f'Training completed for {target}.')
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
    logging.info(f'RMSE for {target}: {rmse}')
    
    log_memory_usage()
    return model

# Train model in parallel using joblib
def train_model(final_df):
    logging.info('Starting model training...')
    
    features = ['Prev_LEAFC', 'Prev_LEAFN', 'Prev_H2OSNO', 'Prev_H2OSOI',
                'FSDS_x', 'TBOT_x', 'QBOT_x', 'WIND_x', 'PSRF_x', 'PRECTmms_x',  # Current day forcing
                'FSDS_y', 'TBOT_y', 'QBOT_y', 'WIND_y', 'PSRF_y', 'PRECTmms_y']  # Previous day forcing
    
    targets = ['LEAFC', 'LEAFN', 'H2OSNO', 'H2OSOI']
    
    X = final_df[features]
    y = final_df[targets]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
    
    models = Parallel(n_jobs=4)(delayed(train_single_model)(X_train_scaled, y_train, X_test_scaled, y_test, target) for target in targets)
    
    logging.info('Model training completed.')
    return models

# Main execution
if __name__ == "__main__":
    logging.info('Starting script...')
    
    try:
        forcing_data, state_data = load_data()
        final_df = prepare_data(forcing_data, state_data)
        models = train_model(final_df)
    except Exception as e:
        logging.error(f'Error occurred: {str(e)}', exc_info=True)
    
    logging.info('Script completed.')
