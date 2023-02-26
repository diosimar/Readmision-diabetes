# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:22:31 2023

@author: diosimarcardoza
"""

from sklearn.model_selection import  cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from src.utils import  update_model, save_simple_metrics_report, get_model_performance_test_set
import logging
import sys
import numpy as np
import pandas as pd

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logger.info('Loading Data by training...')

data_train = pd.read_csv('./data/data_train.csv', na_values='?', low_memory=False)

logger.info('Loading Data by testing..')

data_test = pd.read_csv('./data/data_test.csv', na_values='?', low_memory=False)

logger.info('Loading model...')

model = GradientBoostingClassifier()

logger.info('Seraparating feactures and objetive varible....')

X_train = data_train.drop(['Readmitted'], axis= 1)
y_train = data_train['Readmitted']

X_test = data_test.drop(['Readmitted'], axis= 1)
y_test = data_test['Readmitted']

logger.info('Setting Hyperparameter to tune')

param_tuning = {
               "n_estimators" : [5,50],
               "max_depth":[3,9],
               "learning_rate":[0.01,0.1,1,]
              }


grid_search = GridSearchCV(model, param_grid= param_tuning, scoring='accuracy', cv=3)


logger.info('Starting grid search...')
grid_search.fit(X_train, y_train)

logger.info('Cross validating with best model...')
final_result = cross_validate(grid_search.best_estimator_, X_train, y_train, return_train_score=True, cv=5)

train_score = np.mean(final_result['train_score'])
test_score = np.mean(final_result['test_score'])

logger.info(f'Train Score: {train_score}')
logger.info(f'Test Score: {test_score}')




model = GradientBoostingClassifier(**grid_search.best_estimator_.get_params())

# guardar el mejor modelo entrenado
import pickle
filename = 'Modelo/finalized_model.pkl'
pickle.dump(model, open(filename, 'wb'))

logger.info('Training Finished')

