# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:45:14 2023

@author: diosimarcardoza
"""

from src.utils import  check_label, AgrupadorDeClases, Homologar
import logging
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer


import warnings
warnings.filterwarnings("ignore")


# cargue de  data set desde  raiz del proyecto formato csv
data = pd.read_csv('./data/diabetic_data.csv', na_values='?', low_memory=False)
IDs_mapping= pd.read_csv('./data/IDs_mapping.csv', na_values='?', low_memory=False)


# Encontrar la posición de la fila que contiene registros NaN
posicion = np.where(IDs_mapping.isna().all(axis=1))
# se divide el dataset de mapping en 3, con el objetivo de identificar los merge`s` completos 

admission_type_id = IDs_mapping.iloc[:posicion[0][0], : ]
admission_type_id.columns = ['admission_type_id' , 'admission_type']
admission_type_id['admission_type_id'] = admission_type_id['admission_type_id'].astype(int)
admission_type_id['admission_type'] = admission_type_id['admission_type'].str.strip()
 
discharge_disposition_id = IDs_mapping.iloc[posicion[0][0]+2:posicion[0][1], : ]
discharge_disposition_id.columns = ['discharge_disposition_id',	'discharge_disposition']
discharge_disposition_id['discharge_disposition_id']  = discharge_disposition_id['discharge_disposition_id'].astype(int)
discharge_disposition_id['discharge_disposition']  = discharge_disposition_id['discharge_disposition'].str.strip()


admission_source_id = IDs_mapping.iloc[posicion[0][1]+2:, : ]
admission_source_id.columns = ['admission_source_id','admission_source']
admission_source_id['admission_source_id'] = admission_source_id['admission_source_id'].astype(int)
admission_source_id['admission_source'] = admission_source_id['admission_source'].str.strip()

###############Merge entre el df inicial y las tablas homologadoras de ids#############################################
df = pd.merge(data, admission_type_id, on='admission_type_id')
df = pd.merge(df, discharge_disposition_id, on='discharge_disposition_id')
df = pd.merge(df, admission_source_id, on='admission_source_id')

cols = df.columns.difference(['admission_type_id', 'discharge_disposition_id', 'admission_source_id'])

df= df[cols]




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

### eliminar inconveniente de valor no registrado en la columna discharge_disposition
df.drop(df[df['discharge_disposition'] == 'Unknown/Invalid'].index, inplace = True)
df.reset_index(inplace = True, drop = True)

### eliminar inconveniente de valor no registrado en la columna admission_source_
df.drop(df[df['admission_source'] == 'Unknown/Invalid'].index, inplace = True)
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
    '''
    Parameters
    ----------
    data : Dataframe  de pandas con la informacion a analizar.
    cols : lista de numpy
        se especifican las  columnas  que se encuentran dentro del dataframe y que seran objeto de 
        tranformacion por medio de una homologacion sobre  los codigos CIE-10 de diagnostico (estandarizacion por la OMS).

    Returns
    -------
    data : dataframe de pandas
        Al final se obtiene un df con toda la informacion tranformada.

    '''
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
df['readmitted'] = df.readmitted.apply(check_label) 

##### split de dataset para entrenamiento y validacion 
from sklearn.model_selection import train_test_split
X = df.drop(['readmitted'], axis=1)
y = df['readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## hacer dentro del pipeline para gestionar las transformaciones de la reduccion de calses y la homologacion de los rangos de edad

#### reduccion de caracteristicas 
columns_to_reduce = ['discharge_disposition', 'admission_source']
   
# diccionario para mapear
ordinal_mappings =  {'[0-10)':0, '[10-20)':1, '[20-30)':2, '[30-40)':3, '[40-50)':4, '[50-60)':5,
       '[60-70)':6, '[70-80)':7, '[80-90)':8, '[90-100)':9  }


from sklearn.pipeline import make_pipeline

preprocessing_pipeline = make_pipeline(
    AgrupadorDeClases(cols= columns_to_reduce, n_grupos = 26), 
    Homologar(col_name='age', dictionary = ordinal_mappings))
    
X_train_prep = preprocessing_pipeline.fit_transform(X_train)
X_test_prep  = preprocessing_pipeline.transform(X_test)

X_test_prep['admission_source'] = X_test_prep['admission_source'].replace('Sick Baby', 'otro')




le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)



### escalar variables continuas 
Scale_features =  list((X_train_prep.select_dtypes(np.number)).columns)

Scale_transformer = Pipeline(steps=[
    ('Scaling', MinMaxScaler())
])

### encoder para variables categoricas ordinales
Ordi_features = list((X_train_prep.select_dtypes('O')).columns)
Ordi_features.remove('gender')

Ordi_transformer = Pipeline(steps=[
    ('Ordi', OrdinalEncoder())
])
### encoder para variables categoricas no ordinales
NonO_features = ['gender']

NonO_transformer = Pipeline(steps=[
    ('Non-O', OneHotEncoder())
])


Preprocessor = ColumnTransformer(transformers=[
    ('Scale', Scale_transformer, Scale_features),
    ('Ordinal', Ordi_transformer, Ordi_features),
    ('Non-Ordinal', NonO_transformer, NonO_features)
], remainder = 'passthrough')
    
clf = Pipeline(steps=[('preprocessor', Preprocessor)])

trainX_df = clf.fit_transform(X_train)
testX_df = clf.fit_transform(X_test)

# extraer los nombres de columna del encoder step del pipeline

ohe_cols = clf.named_steps['preprocessor'].transformers_[2][1]\
    .named_steps['Non-O'].get_feature_names_out(NonO_features)
ohe_cols = [x for x in ohe_cols]


cols = [y for x in [Scale_features, Ordi_features, ohe_cols] for y in x]
cols


transformed_x_train = pd.DataFrame(trainX_df, columns= cols)
transformed_x_test = pd.DataFrame(testX_df, columns= cols)


### seleccion de  caracteristicas 



# Crear un modelo de regresión lineal
modelo = LinearRegression()

# Crear un objeto RFE
rfe = RFE(modelo, n_features_to_select=20)

# Aplicar RFE a los datos
X_nuevo = rfe.fit_transform(transformed_x_train,y_train)

# Obtenemos los nombres de las características seleccionadas
caracteristicas_seleccionadas = rfe.get_support()


nombres_caracteristicas = transformed_x_train.columns
nombres_caracteristicas_seleccionadas = nombres_caracteristicas[caracteristicas_seleccionadas]


### dataframe final 
data_train = pd.concat([transformed_x_train[nombres_caracteristicas_seleccionadas],pd.DataFrame(y_train, columns = ['Readmitted'])],axis=1)  
data_test = pd.concat([transformed_x_test[nombres_caracteristicas_seleccionadas] ,pd.DataFrame(y_test, columns = ['Readmitted'])],axis=1)  

data_train.to_csv('./data/data_train.csv', index=False)
data_test.to_csv('./data/data_test.csv', index=False)