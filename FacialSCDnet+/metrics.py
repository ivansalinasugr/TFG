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

def r2(predicciones, objetivos):
    # Convertir a tensores de PyTorch si no lo son
    if not isinstance(predicciones, torch.Tensor):
        predicciones = torch.tensor(predicciones, dtype=torch.float32)
    if not isinstance(objetivos, torch.Tensor):
        objetivos = torch.tensor(objetivos, dtype=torch.float32)

    # Asegurarse de que las dimensiones sean compatibles
    if predicciones.shape != objetivos.shape:
        raise ValueError("Las predicciones y los objetivos deben tener las mismas dimensiones")

    # Calcular la media de los objetivos
    media_objetivos = torch.mean(objetivos)

    # Calcular la suma de los residuos al cuadrado (SS_res)
    ss_res = torch.sum((objetivos - predicciones) ** 2)

    # Calcular la suma total de los cuadrados (SS_tot)
    ss_tot = torch.sum((objetivos - media_objetivos) ** 2)

    # Calcular el coeficiente de determinación R^2
    r2 = 1 - ss_res / ss_tot

    return r2, r2, r2, r2, r2, r2, r2


def r2_ajustado(predicciones, objetivos):
    # Convertir a tensores de PyTorch si no lo son
    if not isinstance(predicciones, torch.Tensor):
        predicciones = torch.tensor(predicciones, dtype=torch.float32)
    if not isinstance(objetivos, torch.Tensor):
        objetivos = torch.tensor(objetivos, dtype=torch.float32)

    # Asegurarse de que las dimensiones sean compatibles
    if predicciones.shape != objetivos.shape:
        raise ValueError("Las predicciones y los objetivos deben tener las mismas dimensiones")

    # Calcular la media de los objetivos
    media_objetivos = torch.mean(objetivos)

    # Calcular la suma de los residuos al cuadrado (SS_res)
    ss_res = torch.sum((objetivos - predicciones) ** 2)

    # Calcular la suma total de los cuadrados (SS_tot)
    ss_tot = torch.sum((objetivos - media_objetivos) ** 2)

    # Calcular el coeficiente de determinación R^2
    r2 = 1 - ss_res / ss_tot

    # Número de observaciones
    n_observaciones = len(objetivos)

    # Calcular el coeficiente de determinación R^2 ajustado
    r2_ajustado = 1 - ((1 - r2) * (n_observaciones - 1) / (n_observaciones - 25088 - 1))

    return r2_ajustado, r2_ajustado, r2_ajustado, r2_ajustado, r2_ajustado, r2_ajustado, r2_ajustado

# def mean_squared_log_error(y_true, y_pred):
#     msle = torch.mean(torch.log1p(y_true) - torch.log1p(y_pred))
#     return msle

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
    
    return value*100, std*100, median*100, min_value*100, perc_90*100, perc_95*100, perc_99*100


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
    'distortloss': distortloss,
    'r2': r2,
    'r2_ajustado': r2_ajustado
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


