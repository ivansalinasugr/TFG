import random
import pandas as pd
from sklearn.model_selection import train_test_split

# parser = argparse.ArgumentParser(description='This program randomizes the training and testing sets used for the DL model')
# parser.add_argument('--data', type=str, help='Path to a video or a sequence of image.', default="./SCD_labels.h5")

# args = parser.parse_args()

# scd_df=pd.read_hdf(args.data,'data')

scd_df=pd.read_csv("labels.csv")
total_rows=len(scd_df.index) 
#Define percentages for data split into train, validation, and test
tset=int(0.2*total_rows) 

# Seleccionar 30 sujetos aleatorios
random_subjects = random.sample(range(5, 176, 5), 6)
print('Chosen subjects:', random_subjects)
# Crear una lista para almacenar el número total de filas de prueba para cada sujeto seleccionado
test_nrows_list = []
# Calcular el número total de filas de prueba para cada sujeto seleccionado y restarlas de tset
for filter_subject in random_subjects:
    print('Processing subject:', filter_subject)
    is_subject = (scd_df.Subject == filter_subject)
    test_nrows = is_subject.sum()
    test_nrows_list.append(test_nrows)
# Restar el número total de filas de prueba para todos los sujetos seleccionados de tset
tset = tset - sum(test_nrows_list)

#Split the dataset
scd_df_filter=scd_df[~is_subject]
scd_test=scd_df[is_subject].reset_index(drop=True)  
filter_nrows=len(scd_df_filter.index) 
#Shuffle the remaining  
scd_df_shuffle = scd_df_filter.sample(frac=1).reset_index(drop=True) 

#Merge random subject with randomly selected to meet tset percentage for the data split
scd_df_test = pd.concat([scd_test, scd_df_shuffle.iloc[:tset,:]], ignore_index=True)
scd_df_train=scd_df_shuffle.iloc[tset:,:].reset_index(drop=True)  

#Reshuffle test set
scd_df_test= scd_df_test.sample(frac=1).reset_index(drop=True)  
 

sub_sample_size=-1
X=scd_df_train.loc[:, scd_df_train.columns != 'Distance'][:sub_sample_size]
y=scd_df_train['Distance'].values[:sub_sample_size]
print(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2 , random_state=42)

print(X_train['Path'].values )
print(y_train)
scd_df_train.to_csv('./train_labels.csv',index = None)
scd_df_test.to_csv('./test_labels.csv',index = None)

