# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 11:45:14 2023

@author: diosimarcardoza
"""

from src.utils import plot_count , check_label, barplot_per_classes, kdeplot_per_classes , boxplot_per_classes
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

# como primer paso del procesamiento de datos 