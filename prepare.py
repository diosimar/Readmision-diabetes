# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:45:14 2023

@author: diosimarcardoza
"""

from src.utils import plot_count , check_label, barplot_per_classes, kdeplot_per_classes \
      , boxplot_per_classes,eliminar_outliers, AgrupadorDeClases, Homologar
from io import StringIO
import sys
import logging
import pandas as pd
import numpy as np
from prettytable import PrettyTable
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly 
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")


# cargue de  data set desde  raiz del proyecto formato csv
df = pd.read_csv('./data/diabetic_data.csv', na_values='?', low_memory=False)
IDs_mapping= pd.read_csv('./data/IDs_mapping.csv', na_values='?', low_memory=False)

# en primer lugar, se eliminan columnas con datos perdidos superior al 15%
# ademas, se eliminan las columnas no relevantes para el analisis
cols=['encounter_id' ,'patient_nbr', 'payer_code', 'weight', 'medical_specialty' , 'citoglipton', 'examide']
df.drop(columns= cols,  inplace = True)

## imputacion de datos 

def imputar_categoricas(datos):
    for columna in datos.columns:
        if datos[columna].dtype == "object":
            datos[columna].fillna(datos[columna].value_counts().index[0], inplace=True)
    return datos

df = imputar_categoricas(df)


### eliminar inconveniente de valor no registrado en la columna genero
df.drop(df[df['gender'] == 'Unknown/Invalid'].index, inplace = True)
df.reset_index(inplace = True, drop = True)


# creacion de nuevas caracteristicas.

df['visits_sum'] = df.apply( lambda x: x['number_emergency'] + x['number_outpatient'] + x['number_inpatient'], axis=1)

all_medicaments = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide','glimepiride', 
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
    'insulin','glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]

col_name = 'number_medicaments_changes'
df[col_name] = 0

for medicament in all_medicaments:
    df[col_name] = df.apply(
        lambda x: x[col_name] + 1 if x[medicament] not in ['No', 'Steady'] else x[col_name], axis=1
    )
    

df['number_medicaments'] = df[all_medicaments].apply(
    lambda y: y.apply(lambda x: np.sum(0 if x == 'No' else 1)), axis=1
).apply(np.sum, axis=1)

## homologacion de los campos con codigos CIE_10 de diagnosticos 
def map_diagnosis(data, cols):
    for col in cols:
        data.loc[(data[col].str.contains("V")) | (data[col].str.contains("E")), col] = -1
        data[col] = data[col].astype(np.float16)

    for col in cols:
        data["temp_diag"] = np.nan
        data.loc[(data[col]>=390) & (data[col]<=459) | (data[col]==785), "temp_diag"] = "Circulatory"
        data.loc[(data[col]>=460) & (data[col]<=519) | (data[col]==786), "temp_diag"] = "Respiratory"
        data.loc[(data[col]>=520) & (data[col]<=579) | (data[col]==787), "temp_diag"] = "Digestive"
        data.loc[(data[col]>=250) & (data[col]<251), "temp_diag"] = "Diabetes"
        data.loc[(data[col]>=800) & (data[col]<=999), "temp_diag"] = "Injury"
        data.loc[(data[col]>=710) & (data[col]<=739), "temp_diag"] = "Muscoloskeletal"
        data.loc[(data[col]>=580) & (data[col]<=629) | (data[col] == 788), "temp_diag"] = "Genitourinary"
        data.loc[(data[col]>=140) & (data[col]<=239), "temp_diag"] = "Neoplasms"

        data["temp_diag"] = data["temp_diag"].fillna("Other")
        data[col] = data["temp_diag"]
        data = data.drop("temp_diag", axis=1)

    return data

df = map_diagnosis(df,["diag_1","diag_2","diag_3"])

## codificar la variable objetivo 
df['label'] = df.readmitted.apply(check_label) 

##### split de dataset para entrenamiento y validacion 
from sklearn.model_selection import train_test_split
X = df.drop(['readmitted', 'label'], axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



## hacer dentro del pipeline 

from sklearn.pipeline import make_pipeline



### ajuste para homologacion de categoria de edad 

ordinal_mappings =  {'[0-10)':0, '[10-20)':1, '[20-30)':2, '[30-40)':3, '[40-50)':4, '[50-60)':5,
       '[60-70)':6, '[70-80)':7, '[80-90)':8, '[90-100)':9  }

homologador = Homologar(col_name='age', dictionary = ordinal_mappings)
df = homologador.fit_transform(df)



#### reduccion de calses  
columns_to_reduce = ['discharge_disposition_id', 'admission_source_id']
    
# crear una instancia del transformador y ajustarlo a los datos de entrenamiento
agrupador = AgrupadorDeClases(cols= columns_to_reduce, n_grupos=2)
df = agrupador.fit_transform(df)

###################################################################################################
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()

categorical_features =['race', 'gender', 'age',
       'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
       'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
       'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin',
       'glyburide-metformin', 'change', 'diabetesMed'] 

for i in categorical_features:
    df_[i] = le.fit_transform(df_[i])
