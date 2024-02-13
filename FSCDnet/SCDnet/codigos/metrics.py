import torch

def mean_absolute_error(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred))
    percentil_99 = torch.kthvalue(torch.abs(y_true - y_pred), int(len(y_true) * 0.99)).values
    mediana = torch.median(torch.abs(y_true - y_pred))
    desviacion_tipica = torch.std(y_true - y_pred)
    return mae, percentil_99, mediana, desviacion_tipica

def mean_squared_error(y_true, y_pred):
    mse = torch.mean((y_true - y_pred)**2)
    percentil_99 = torch.kthvalue(torch.abs(y_true - y_pred), int(len(y_true) * 0.99)).values
    mediana = torch.median(torch.abs(y_true - y_pred))
    desviacion_tipica = torch.std(y_true - y_pred)
    return mse, percentil_99, mediana, desviacion_tipica

def root_mean_squared_error(y_true, y_pred):
    rmse = torch.sqrt(torch.mean((y_true - y_pred)**2))
    percentil_99 = torch.kthvalue(torch.abs(y_true - y_pred), int(len(y_true) * 0.99)).values
    mediana = torch.median(torch.abs(y_true - y_pred))
    desviacion_tipica = torch.std(y_true - y_pred)
    return rmse, percentil_99, mediana, desviacion_tipica

def median_absolute_error(y_true, y_pred):
    medianae = torch.median(torch.abs(y_true - y_pred))
    percentil_99 = torch.kthvalue(torch.abs(y_true - y_pred), int(len(y_true) * 0.99)).values
    mediana = torch.median(torch.abs(y_true - y_pred))
    desviacion_tipica = torch.std(y_true - y_pred)
    return medianae, percentil_99, mediana, desviacion_tipica

def r2_score(y_true, y_pred):
    SSR = torch.sum((y_pred - torch.mean(y_true))**2)
    SST = torch.sum((y_true - torch.mean(y_true))**2)
    percentil_99 = torch.kthvalue(torch.abs(y_true - y_pred), int(len(y_true) * 0.99)).values
    mediana = torch.median(torch.abs(y_true - y_pred))
    desviacion_tipica = torch.std(y_true - y_pred)
    return 1 - SSR / SST, percentil_99, mediana, desviacion_tipica

def mean_relative_error(y_true, y_pred):
    mre = torch.mean(torch.abs((y_true - y_pred) / y_true))
    percentil_99 = torch.kthvalue(torch.abs(y_true - y_pred), int(len(y_true) * 0.99)).values
    mediana = torch.median(torch.abs(y_true - y_pred))
    desviacion_tipica = torch.std(y_true - y_pred)
    return mre, percentil_99, mediana, desviacion_tipica

def mean_squared_log_error(y_true, y_pred):
    msle = torch.mean(torch.log1p(y_true) - torch.log1p(y_pred))
    percentil_99 = torch.kthvalue(torch.abs(y_true - y_pred), int(len(y_true) * 0.99)).values
    mediana = torch.median(torch.abs(y_true - y_pred))
    desviacion_tipica = torch.std(y_true - y_pred)
    return msle, percentil_99, mediana, desviacion_tipica


# Dictionary mapping metric names to their corresponding functions
metric_functions = {
    'mae': mean_absolute_error,
    'mse': mean_squared_error,
    'rmse': root_mean_squared_error,
    'medae': median_absolute_error,
    'r2': r2_score,
    'mre': mean_relative_error,
    'msle': mean_squared_log_error
}

# Function for calculate metrics
def calculate_metrics(metrics, y_true, y_pred, prefix=''):
    results = {}
    for metric_name in metrics:
        if metric_name in metric_functions:
            # Call the corresponding function to calculate the metric
            result = metric_functions[metric_name](y_true, y_pred)
            # Save results in a dicctionary
            results[metric_name] = {
                f'{prefix}{metric_name}_value': result[0].item(),  # Value
                f'{prefix}{metric_name}_perc99': result[1].item(),  # 99 Percentile
                f'{prefix}{metric_name}_median': result[2].item(),  # Median
                f'{prefix}{metric_name}_std': result[3].item()  # Standard desviation
            }
        else:
            print(f"Warning: Metric '{metric_name}' not recognized")
            results[metric_name] = None
    return results