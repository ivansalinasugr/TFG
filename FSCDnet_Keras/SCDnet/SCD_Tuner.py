import hyperopt
from hyperopt import fmin, hp,tpe, Trials
from hyperopt.mongoexp import MongoTrials
import numpy as np
from SCD_Main_Tuner import hyper_objective 
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

if __name__ == '__main__': 
    name = 'SCD_main'
    version = 4

    fspace = {
        'img_set':'real',#fullsynth,real',
        'focal': 35, 
        'n_epochs':150,
        'batch_size': 64,   
        #TODO: Verify synth best learning rates                     
        #'initial_learning_rate':hp.loguniform('initial_learning_rate', np.log(0.00009),np.log(0.003)),       
        'initial_learning_rate':hp.loguniform('initial_learning_rate', np.log(0.0000001),np.log(0.0001)),       
        'normalize':False,
        'loss': 'distortloss',#hp.choice('loss',['mae','distortmae']), #hp.choice('loss',['mae','huber','persploss']),
        #'optimizer': hp.choice('optimizer',['adam','adagrad']),
        'backbone':'resnet',# vgg, resnet, inception
        'lastConv':hp.choice('lastconv',[True,False]),
        #'size_fc2':hp.choice('size_fc2',[4096,2048]),
       #'dp': hp.choice('dp',
        #[
        #    {'with_dropout': False,'dropout':0},
        #    {'with_dropout': True,'dropout':  hp.choice('dropout',np.linspace(0.3, 0.5, 2, dtype=float)),'with_batchnormalization':False},
        #    {'with_dropout': True,'dropout':  hp.choice('dropoutbn',np.linspace(0.3, 0.5, 2, dtype=float)),'with_batchnormalization':True}
        #]),
        'patience': hp.choice('patience', [2,3]),
        'early_stop': hp.choice('early_stop', [4,6]), 
    }
    #TODO:
    root_path=os.path.abspath("/mnt/homeGPU/ebermejo/SCDForensics/results_"+str(fspace['backbone'])+"_"+str(fspace['focal']))
    # trials = MongoTrials('mongo://localhost:1234/hyperopt/jobs',
    #                           exp_key='%s_v%d' % (name, version))
    #trials = Trials()	
    trials = joblib.load(root_path+'/artifacts/hyperopt_trials.pkl')
    argmin = fmin(fn=hyper_objective,space=fspace,algo=tpe.suggest,max_evals=80,trials=trials,verbose=2, show_progressbar=False, rstate=np.random.RandomState(1))
    joblib.dump(trials, root_path+'/artifacts/hyperopt_trials1.pkl')
    res_time=trials.best_trial['result']['eval_time']
    best_result = trials.best_trial['result']['loss']
    print('best result = ', best_result, '\nparams = ', argmin, '\nelapsed time = ',res_time)
