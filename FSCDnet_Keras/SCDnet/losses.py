
import tensorflow.keras.backend as K
import tensorflow as tf

def L1(y_true, y_pred): # LAE
    return K.sum( K.abs(y_true - y_pred))

def L2(y_true, y_pred): #LSE
    return K.sum( K.abs(y_true - y_pred)**2)

def logloss(yTrue,yPred): 
    return K.sum(K.log(yTrue) - K.log(yPred))

def distortloss(y_true, y_pred): 
    true = (1/(1 + y_true/12.6572))
    pred = (1/(1 + y_pred/12.6572))
    return K.sum( K.abs(true - pred))

def distortmae(y_true, y_pred): 
    true = 100*(1/(1 + y_true/12.6572)) #We scale distortion to balance mae units 
    pred = 100*(1/(1 + y_pred/12.6572))
    w1=0.4
    mae= w1*K.sum( K.abs(y_true - y_pred))
    distort=(1-w1)* K.sum( K.abs(true - pred))
    return  mae + distort

 
