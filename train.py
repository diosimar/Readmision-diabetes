# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:22:31 2023

@author: diosimarcardoza
"""

from sklearn.model_selection import  cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

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

model = Pipeline([
    ('core_model', GradientBoostingClassifier())
])

logger.info('Seraparating feactures and objetive varible....')

X_train = data_train.drop(['Readmitted'], axis= 1)
y_train = data_train['Readmitted']

X_test = data_test.drop(['Readmitted'], axis= 1)
y_test = data_test['Readmitted']

logger.info('Setting Hyperparameter to tune')
param_tuning = {'core_model__n_estimators':range(20,301,20)}

grid_search = GridSearchCV(model, param_grid= param_tuning, scoring='r2', cv=5)


logger.info('Starting grid search...')
grid_search.fit(X_train, y_train)

logger.info('Cross validating with best model...')
final_result = cross_validate(grid_search.best_estimator_, X_train, y_train, return_train_score=True, cv=5)

train_score = np.mean(final_result['train_score'])
test_score = np.mean(final_result['test_score'])
assert train_score > 0.7
assert test_score > 0.65

logger.info(f'Train Score: {train_score}')
logger.info(f'Test Score: {test_score}')

logger.info('Updating model...')
update_model(grid_search.best_estimator_)

logger.info('Generating model report...')
validation_score = grid_search.best_estimator_.score(X_test, y_test)
save_simple_metrics_report(train_score, test_score, validation_score, grid_search.best_estimator_)

y_test_pred = grid_search.best_estimator_.predict(X_test)
get_model_performance_test_set(y_test, y_test_pred)

logger.info('Training Finished')














LR = LogisticRegression()

LR.fit(transformed_x_train[nombres_caracteristicas_seleccionadas],y_train)
LR.score(transformed_x_test[nombres_caracteristicas_seleccionadas] ,y_test)
  

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()

RF.fit(transformed_x_train[nombres_caracteristicas_seleccionadas],y_train)
RF.score(transformed_x_test[nombres_caracteristicas_seleccionadas] ,y_test)