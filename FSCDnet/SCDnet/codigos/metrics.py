import torch
import numpy as np

# def mean_squared_error(y_true, y_pred):
#     mse = torch.mean((y_true - y_pred)**2)
#     return mse

# def root_mean_squared_error(y_true, y_pred):
#     rmse = torch.sqrt(torch.mean((y_true - y_pred)**2))
#     return rmse

# def median_absolute_error(y_true, y_pred):
#     medianae = torch.median(torch.abs(y_true - y_pred))
#     return medianae

# def r2_score(y_true, y_pred):
#     SSR = torch.sum((y_pred - torch.mean(y_true))**2)
#     SST = torch.sum((y_true - torch.mean(y_true))**2)
#     return 1 - SSR / SST

# def mean_squared_log_error(y_true, y_pred):
#     msle = torch.mean(torch.log1p(y_true) - torch.log1p(y_pred))
#     return msle

import torch
import numpy as np

def mean_absolute_error(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred))
    std = torch.std(torch.abs(y_true - y_pred))
    median = torch.median(torch.abs(y_true - y_pred))
    min_value = torch.min(torch.abs(y_true - y_pred))
    perc_90 = torch.quantile(torch.abs(y_true - y_pred), 0.90)
    perc_95 = torch.quantile(torch.abs(y_true - y_pred), 0.95)
    perc_99 = torch.quantile(torch.abs(y_true - y_pred), 0.99)
    
    return mae, std, median, min_value, perc_90, perc_95, perc_99

def mean_relative_error(y_true, y_pred):
    mre = torch.mean(torch.abs((y_true - y_pred) / y_true))
    std = torch.std(torch.abs((y_true - y_pred) / y_true))
    median = torch.median(torch.abs((y_true - y_pred) / y_true))
    min_value = torch.min(torch.abs((y_true - y_pred) / y_true))
    perc_90 = torch.quantile(torch.abs((y_true - y_pred) / y_true), 0.90)
    perc_95 = torch.quantile(torch.abs((y_true - y_pred) / y_true), 0.95)
    perc_99 = torch.quantile(torch.abs((y_true - y_pred) / y_true), 0.99)
    
    return mre, std, median, min_value, perc_90, perc_95, perc_99

def distortloss(y_true, y_pred): 
    true = (1/(1 + y_true/12.6572))
    pred = (1/(1 + y_pred/12.6572))
    value = torch.mean(torch.abs(true - pred))
    std = torch.std(torch.abs(true - pred))
    median = torch.median(torch.abs(true - pred))
    min_value = torch.min(torch.abs(true - pred))
    perc_90 = torch.quantile(torch.abs(true - pred), 0.90)
    perc_95 = torch.quantile(torch.abs(true - pred), 0.95)
    perc_99 = torch.quantile(torch.abs(true - pred), 0.99)
    
    return value, std, median, min_value, perc_90, perc_95, perc_99


# Dictionary mapping metric names to their corresponding functions
# metric_functions = {
#     'mae': mean_absolute_error,
#     'mse': mean_squared_error,
#     'rmse': root_mean_squared_error,
#     'medae': median_absolute_error,
#     'r2': r2_score,
#     'mre': mean_relative_error,
#     'msle': mean_squared_log_error,
#     'distortloss': distortloss
# }

metric_functions = {
    'mae': mean_absolute_error,
    'mre': mean_relative_error,
    'distortloss': distortloss
}

# Function for calculate metrics
def calculate_metrics(metrics, y_true, y_pred, prefix=''):
    results = {}
    for metric_name in metrics:
        if metric_name in metric_functions:
            # Call the corresponding function to calculate the metric
            result = metric_functions[metric_name](y_true, y_pred)
            # Save results in a dictionary
            results[metric_name] = {
                f'{prefix}{metric_name}_mean': result[0].item(),  # Mean
                f'{prefix}{metric_name}_std': result[1].item(),  # Standard deviation
                f'{prefix}{metric_name}_median': result[2].item(),  # Median
                f'{prefix}{metric_name}_min': result[3].item(),  # Minimum
                f'{prefix}{metric_name}_perc_90': result[4].item(),  # 90th percentile
                f'{prefix}{metric_name}_perc_95': result[5].item(),  # 95th percentile
                f'{prefix}{metric_name}_perc_99': result[6].item(),  # 99th percentile
            }
        else:
            print(f"Warning: Metric '{metric_name}' not recognized")
            results[metric_name] = None
    return results


