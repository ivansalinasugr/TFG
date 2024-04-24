import random
import pandas as pd
from sklearn.model_selection import train_test_split

# parser = argparse.ArgumentParser(description='This program randomizes the training and testing sets used for the DL model')
# parser.add_argument('--data', type=str, help='Path to a video or a sequence of image.', default="./SCD_labels.h5")

# args = parser.parse_args()

# scd_df=pd.read_hdf(args.data,'data')

random.seed(2020)

scd_df=pd.read_csv("labels.csv")
total_rows=len(scd_df) 
#Define percentages for data split into train, validation, and test
tset=int(0.2*total_rows) 
print("Imágenes totales: " + str(total_rows))
print("Imágenes para train: " + str(total_rows-tset))
print("Imágenes para test: " + str(tset))

# Seleccionar 50 sujetos aleatorios
subjects = scd_df['Subject'].unique()
random_subjects = random.sample(list(subjects), 50)
print('Chosen subjects:', random_subjects)

# Filtrar las filas correspondientes a los sujetos de prueba
test_df = scd_df[scd_df['Subject'].isin(random_subjects)]

# Filtrar las filas restantes para el conjunto de entrenamiento
train_df = scd_df[~scd_df['Subject'].isin(random_subjects)]
train_df = train_df.sample(frac=1).reset_index(drop=True)

test_restante = tset-len(test_df.index)

print("Imágenes de sujetos enteros: " + str(len(test_df)))
print("Imágenes restantes para test: " + str(test_restante))

# Realizar una división estratificada del conjunto de entrenamiento y prueba basada en la columna 'Focal'
train_df, test_df_remaining = train_test_split(train_df, test_size=test_restante/len(train_df.index), stratify=train_df['Focal'])

# Concatenar las filas restantes para el conjunto de prueba
test_df = pd.concat([test_df, test_df_remaining])

train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

print("Imágenes finales para train: " + str(len(train_df)))
print("Imágenes finales para test: " + str(len(test_df)))

# Contar el número de líneas para cada valor único en la columna 'Focal'
focal_counts = test_df['Focal'].value_counts()

print("Número de líneas por valor único en la columna 'Focal':")
print(focal_counts)

train_df.to_csv('train_labels.csv',index = None)
test_df.to_csv('test_labels.csv',index = None)

