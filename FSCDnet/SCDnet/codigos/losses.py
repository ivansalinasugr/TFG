import torch

def L1(y_true, y_pred): # LAE
    return torch.sum(torch.abs(y_true - y_pred))

def L2(y_true, y_pred): #LSE
    return torch.sum((y_true - y_pred)**2)

def logloss(yTrue,yPred): 
    return torch.sum(torch.log(yTrue) - torch.log(yPred))

def distortloss(y_true, y_pred): 
    true = (1/(1 + y_true/12.6572))
    pred = (1/(1 + y_pred/12.6572))
    return torch.sum(torch.abs(true - pred))