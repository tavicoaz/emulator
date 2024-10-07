import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import load
from netCDF4 import Dataset

# Use Agg backend for non-interactive environments
import matplotlib
matplotlib.use('Agg')

# Define the base paths for the data and results
test_data_path = '/work/cmcc/lg07622/work/emulator/data_test/'
model_dir = '/work/cmcc/lg07622/work/emulator/models/'
results_dir = '/work/cmcc/lg07622/work/emulator/results/'

# Create the results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

def load_data(data_path):
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

# Function to load a saved model
def load_model(target):
    model_filename = os.path.join(model_dir, f'{target}_rf_model.joblib')
    if os.path.exists(model_filename):
        print(f"Loading saved model for {target} from {model_filename}...")
        model = load(model_filename)
        return model
    else:
        print(f"Model for {target} not found!")
        return None

# Plotting function for visualization and saving results to disk
def plot_results(y_true, y_pred, target):
    plt.figure(figsize=(15, 8))

    # Predicted vs Actual Plot
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual for {target}')

    # Residuals Plot
    plt.subplot(1, 3, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', lw=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals for {target}')

    # Time Series Plot
    plt.subplot(1, 3, 3)
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Time (days)')
    plt.ylabel('Values')
    plt.title(f'Time Series for {target}')
    plt.legend()

    plt.tight_layout()

    # Save the figure to disk
    plot_filename = os.path.join(results_dir, f'{target}_evaluation_plot.png')
    plt.savefig(plot_filename)
    print(f"Plot for {target} saved to {plot_filename}")
    plt.close()

# Save results to a NetCDF file
def save_to_netcdf(target, dates, y_true, y_pred):
    netcdf_filename = os.path.join(results_dir, f'{target}_results.nc')
    ncfile = Dataset(netcdf_filename, 'w', format='NETCDF4')

    # Create dimensions
    ncfile.createDimension('time', len(dates))

    # Create variables
    time_var = ncfile.createVariable('time', 'f8', ('time',))
    true_var = ncfile.createVariable(f'{target}_true', 'f8', ('time',))
    pred_var = ncfile.createVariable(f'{target}_pred', 'f8', ('time',))

    # Save data
    time_var[:] = np.arange(len(dates))
    true_var[:] = y_true
    pred_var[:] = y_pred

    # Close the NetCDF file
    ncfile.close()
    print(f"Results for {target} saved to {netcdf_filename}")

# Save the evaluation results (RMSE, MAE) to a CSV file
def save_evaluation_results(evaluation_results):
    results_filename = os.path.join(results_dir, 'evaluation_metrics.csv')
    results_df = pd.DataFrame(evaluation_results).T  # Transpose to get metrics as columns
    results_df.to_csv(results_filename)
    print(f"Evaluation metrics saved to {results_filename}")

# Evaluate the models using the test data and plot/save results
def evaluate_models(test_df):
    targets = ['LEAFC', 'LEAFN', 'H2OSNO', 'H2OSOI']
    forcing_columns = ['FSDS', 'TBOT', 'QBOT', 'WIND', 'PSRF', 'PRECTmms']
    prev_state_columns = ['Prev_LEAFC', 'Prev_LEAFN', 'Prev_H2OSNO', 'Prev_H2OSOI']

    # Dictionary to store evaluation results
    evaluation_results = {}

    # Loop through each target state variable
    for target in targets:
        model = load_model(target)
        if model:
            # Prepare test input for the model
            X_test = test_df[forcing_columns + prev_state_columns]
            
            # Ensure the test data has the same columns as the model expects
            if hasattr(model, 'feature_names_in_'):
                X_test = test_df[model.feature_names_in_]

            y_true = test_df[target]
            y_pred = model.predict(X_test)

            # Evaluate predictions
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)

            # Store evaluation metrics
            evaluation_results[target] = {'RMSE': rmse, 'MAE': mae}
            print(f"Evaluation results for {target}: RMSE = {rmse}, MAE = {mae}")

            # Save results to NetCDF file
            save_to_netcdf(target, test_df['Date'], y_true, y_pred)

            # Generate and save plots
            plot_results(y_true, y_pred, target)

    # Save evaluation metrics to CSV
    save_evaluation_results(evaluation_results)

    return evaluation_results

# Main execution
if __name__ == "__main__":
    print("Starting model evaluation...")

    # Load test data (2008)
    forcing_data_test, state_data_test = load_data(test_data_path)
    test_df = prepare_data(forcing_data_test, state_data_test)

    # Evaluate the models using the test data
    evaluation_results = evaluate_models(test_df)

    print("Finished model evaluation.")
