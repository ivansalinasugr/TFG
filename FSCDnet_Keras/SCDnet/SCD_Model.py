
import matplotlib.pyplot as plt
import pandas as pd
import gc,time
#Keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard,LambdaCallback)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

import losses as lo

class SCDModel():
    def __init__(self, hparams): 
        self.input_dim = hparams['image_shape']  # image input dimensions
        self.expandMemory() 
        self.config=hparams  

        if(self.config['load_model']):
            self.model =    tf.keras.models.load_model(self.config['input_model_file'], custom_objects=None, compile=True)
        else:
            self.model = self.create_model()  # model
            if(self.config['load_weight_model']):
                self.load_weights_only()

    def expandMemory(self):
        #gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        #for device in gpu_devices:
        #		tf.config.experimental.set_memory_growth(device, True) 
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def summary(self):
        self.model.summary()

    def getModel(self):
        return self.model
        
    def release(self):
        del self.model
        gc.collect()
        K.clear_session()
 #In case we want to merge the focal lenght into the network, we should separate this function to create another model from scratch
    #def create_merged_model(self):
        #if(self.config['merge_focal'])...
        #fc_scd = Dense(1, activation='relu', name='fc_scd', activity_regularizer=None)(bn)
        #merged = Concatenate(axis=-1)([fc_scd,focal_layer])
        #output_layer = Dense(1, activation='linear')(merged)
        #model = Model(inputs=[input_layer,focal_layer], outputs=output_layer)
 
    def create_model(self):
        input_layer = layers.Input(shape=self.input_dim)
        #focal_layer = layers.Input(shape=(1,))
        if(self.config['backbone']=='vgg'):
            mNET = VGG16(input_shape =self.input_dim, weights='imagenet', input_tensor=input_layer, include_top=False) 
            for l in mNET.layers[:-4]:
                l.trainable = False
            if(self.config['expand_model']):
                x=input_layer
                for l in mNET.layers[1:]:
                    x =l(x)
            else:
                x = mNET(input_layer)
            #Add custom Top to VGG including Ds and BN 
            x = layers.Flatten(name='flatten')(x)
            x = layers.Dense(4096, activation='relu', name='fc1')(x)

            if(self.config['with_dropout']):
                x = layers.Dropout(self.config['dropout'])(x)
            x = layers.Dense(self.config['size_fc2'], activation='relu', name='fc2')(x)
            if(self.config['with_dropout']):
                x = layers.Dropout(self.config['dropout'])(x)
            if(self.config['with_batchnormalization']):
                x = layers.BatchNormalization()(x)

        elif(self.config['backbone']=='resnet'):
            mNET = ResNet50(input_shape =self.input_dim, weights='imagenet', input_tensor=input_layer, include_top=False) 
            for l in mNET.layers[:165]: 
                l.trainable = False 
            x = mNET(input_layer)
            x = layers.Flatten(name='flatten')(x)   
            if(self.config['lastConv']):
                x = layers.Dense(512, activation='relu', name='fc1')(x)
            
            if(self.config['with_batchnormalization']):
                x = layers.BatchNormalization()(x)
                x = layers.Dense(256, activation='relu', name='fc2')(x)
                if(self.config['with_dropout']):
                    x = layers.Dropout(self.config['dropout'])(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dense(128, activation='relu', name='fc3')(x)
                if(self.config['with_dropout']):
                    x = layers.Dropout(self.config['dropout'])(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dense(64, activation='relu', name='fc4')(x)
                if(self.config['with_dropout']):
                    x = layers.Dropout(self.config['dropout'])(x)
                x = layers.BatchNormalization()(x)

        elif(self.config['backbone']=='inception'):
            mNET = InceptionV3(input_shape =self.input_dim, weights='imagenet', input_tensor=input_layer, include_top=False) 
            for l in mNET.layers:
                l.trainable = False  
            x = mNET(input_layer)
            x = layers.Flatten(name='flatten')(x)
          #  x = layers.Dense(4096, activation='relu', name='fc1')(x)

        output_layer = layers.Dense(1, activation='linear', name='predictions')(x)

        model = Model(inputs=[input_layer], outputs=output_layer)
        model.summary()

        loss=self.config['loss']
        if(loss=='huber'):
            loss=tf.keras.losses.Huber(delta=1.5)
        metrics=self.config['metrics']
        metrics=['mean_absolute_error',lo.distortloss]

        optimizer=optimizers.get(self.config['optimizer'])   
        if(loss=='distortloss'):
            loss=lo.distortloss 
        if(loss=='distortmae'):
            loss=lo.distortmae
            

        #model = tf.keras.utils.multi_gpu_model(model, gpus=2)
        model.compile(optimizer=type(optimizer)(lr=self.config['initial_learning_rate']), loss=loss, metrics=metrics)
        return model

    #def result_dir(self):
    #    result_path = self.config['root']+"results/"
    #    result_path = result_path + "_db"+self.config['name'] \
    #                + "_ep"+repr(self.config["n_epochs"]) \
    #                + "_do"+repr(self.config["dropout"]) \
    #                + "_bz"+repr(self.config["batch_size"]) + "/"
    #    if not os.path.exists(result_path):
    #        os.makedirs(result_path)
    #    return result_path


    def train(self, train_data, val_data, plot_results=False):
        'Trains data on generators'
        print("Starting training")
        # reduces learning rate if no improvement are seen
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                    patience=self.config['patience'],
                                                    verbose=1,
                                                    factor=self.config['learning_rate_drop'],
                                                    min_lr=self.config['minimum_learning_rate'])
        # stop training if no improvements are seen
        early_stop = EarlyStopping(monitor="val_loss",
                                   mode="min",
                                   patience=self.config['early_stop'],
                                   restore_best_weights=True)
        # saves model weights to file
        checkpoint = ModelCheckpoint(self.config['model_file'],
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     save_weights_only=True)

        # train on data
        #STEP_SIZE_TRAIN=(train_data.n//train_data.batch_size)/150
        #STEP_SIZE_VAL=(val_data.n//val_data.batch_size)/150
        #print(train_data.n,STEP_SIZE_TRAIN,val_data.n,STEP_SIZE_VAL)
        #history = self.model.fit_generator(generator=train_data,
        #                                  steps_per_epoch=STEP_SIZE_TRAIN,
        #                                   validation_data=val_data,
        #                                   validation_steps =STEP_SIZE_VAL,
        #                                   epochs=self.config['n_epochs'],
        #                                   callbacks=[learning_rate_reduction, early_stop, checkpoint],
        #                                   verbose=2,
        #                                   )
        history = self.model.fit_generator(generator=train_data,
                                           validation_data=val_data,
                                           epochs=self.config['n_epochs'],
                                           steps_per_epoch=len(train_data),
                                           validation_steps =len(val_data),
                                           callbacks=[learning_rate_reduction, early_stop, checkpoint],
                                           verbose=1,
                                           )
        #print(history.history) 
        #print(history.history.keys())
        #print("Evaluating val_data again")
        #val_score = self.model.evaluate_generator(val_data, verbose=0)

        # plot training history
        if plot_results:
            fig, ax = plt.subplots(2, 1, figsize=(6, 6))
            ax[0].plot(history.history['loss'], label="TrainLoss")
            ax[0].plot(history.history['val_loss'], label="ValLoss")
            ax[0].legend(loc='best', shadow=True)
            ax[1].plot(history.history[self.config['metrics'][0]], label="TrainMae")
            ax[1].plot(history.history['val_mean_absolute_error'], label="ValMae")
            ax[1].legend(loc='best', shadow=True)
            plt.show()

        #train_metrics=[]
        #for m in self.config['metrics']:
        #    train_metrics.append(history.history[m])
        #print('Val loss:', val_score[0])
        #print('Val mse:', val_score[1])
        return history.history

    def evaluate(self, test_data, test_label,load_weights=False):
        'Create basic file submit'
        if(load_weights):
            self.load_weights_only()#self.model.load_weights(self.config['input_model_file'])
        # predict on data
        results = self.model.evaluate_generator(test_data, test_label, verbose=0)
        return results

    def load_weights_only(self):
        self.model.load_weights(self.config['input_model_file'])

    def predict(self, test_data, load_weights=False):
        'Create basic file submit'
        if(load_weights):
            self.load_weights_only()#    self.model.load_weights(self.config['input_model_file'])

        # predict on data
        results = self.model.predict_generator(test_data, verbose=0)
        # binarize prediction
        # rbin = np.where(results > 0.5, 1, 0)
        # save results to dataframe
        #results_to_save = pd.DataFrame({"id": test_data.images_paths,
        #                                "label": results[:,0]
        #                                })
        #results_to_save["id"] = results_to_save["id"].apply(lambda x: x.replace("../input/test/", "").replace(".tif", ""))
        # create submission file
        #results_to_save.to_csv("./submission.csv", index=False)
        return results

    def predictVal(self,image):
        return self.model.predict(image)
        
    def predictImage(self, image):
        self.model.load_weights(self.config['input_model_file'])
        layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
        layer_name = 'block5_conv3'
        #self.model.summary() 
        layermodel = Model(inputs=self.model.inputs, outputs=layer_dict[layer_name].output)
        #layermodel = Model(inputs=self.model.inputs, outputs=self.model.get_layer('vgg16').output)
        results = layermodel.predict(image)
        # i=1
        # for layer in self.model.layers:
        #     if 'conv' in layer.name: 
        #         filters, bias= layer.get_weights()
        #         print('Filters Shape: '+ str(filters.shape, )+" " + 'Bias Shape: '+str(bias.shape)+ "<---- layer: "+str(i))
        #         print("-----------")
        #         i=i+1
        return results

    def predictRawImage(self, image):
        self.model.load_weights(self.config['input_model_file'])
        results = self.model.predict(image)
        # i=1
        # for layer in self.model.layers:
        #     if 'conv' in layer.name: 
        #         filters, bias= layer.get_weights()
        #         print('Filters Shape: '+ str(filters.shape, )+" " + 'Bias Shape: '+str(bias.shape)+ "<---- layer: "+str(i))
        #         print("-----------")
        #         i=i+1
        return results
