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
data = pd.read_csv('./data/diabetic_data.csv', na_values='?', low_memory=False)
IDs_mapping= pd.read_csv('./data/IDs_mapping.csv', na_values='?', low_memory=False)

# en primer lugar, se eliminan columnas con datos perdidos superior al 15%
# ademas, se eliminan las columnas no relevantes para el analisis
cols=['encounter_id' ,'patient_nbr', 'payer_code', 'weight', 'medical_specialty' , 'citoglipton', 'examide']
df = data.drop(columns= cols)

## imputacion de datos 


#### identificacion y eliminacion de outliers para las variables continuas 
num_columns = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
df = eliminar_outliers(df, num_columns, 2)

### ajsute de categoria de edad 

ordinal_mappings =  {'[0-10)':0, '[10-20)':1, '[20-30)':2, '[30-40)':3, '[40-50)':4, '[50-60)':5,
       '[60-70)':6, '[70-80)':7, '[80-90)':8, '[90-100)':9  }





homologador = Homologar(col_name='age', dictionary = ordinal_mappings)
df_homologado = homologador.fit_transform(df)

       
#### reduccion de calses  
columns_to_reduce = ['discharge_disposition_id', 'admission_source_id']
    
# crear una instancia del transformador y ajustarlo a los datos de entrenamiento
agrupador = AgrupadorDeClases(cols= columns_to_reduce, n_grupos=2)
X_train = agrupador.fit_transform(df)

## calculo de nuevas caracteristicas
    
'[0-10)' in ordinal_mappings
