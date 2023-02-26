from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from joblib import dump
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt 


def plot_count(feature, title, df, size=1):
    """
    Grafica el plot_count de atributos para cada clase categorica
    
    :param df: dataframe  con la data que  sera graficada en el plot_count.
    :param feature: nombre del atributo a ser graficado en el plot_count.
    :param title: titulo del grafico.
    :param size:  tamaño que  se desea que se expanda la grafica de forma horizontal.
    """
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    #total = float(len(df))
    g = sns.countplot(x = df[feature], order = df[feature].value_counts().index, palette='Set3')
    g.set_title("Resultado por cada categoria de {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}'.format(height),
                ha="center") 
    plt.show()


def check_label(text):
    """
    función que pérmite  categorizar la una variable     
    :param text: variable del dataframe que se va a  categorizar.
    """
    if text == '>30' or text =='<30':
        return 'Si'
    else:
        return 'No'
    


def barplot_per_classes(df, attribute, groupby, title=None, ticks_rotation=0, topn=None, ax=None):
    """
    Grafica el Barplot de atributos para cada clase categorica
    
    :param df: dataframe  con la data que  sera graficada en el barplot.
    :param attribute: nombre del atributo a ser graficado en el barplot.
    :param groupby: nombre del atributo con la clase predictora.
    :param title: titulo del grafico.
    :param ticks_rotation:  rotacion sobre  x-ticks (etiquetas).
    :param topn: top n de clases a ser graficadas en el barplot.
    :param ax: objetos de eje de matplotlib considerado para la grafica.
    """
    uniq_values = df[attribute].value_counts().head(topn).index
    df = df[df[attribute].isin(uniq_values)]
    data = df.groupby(groupby)[attribute].value_counts(normalize=True).rename('porcentaje').mul(100).reset_index()
    sns.barplot(data = data , x = attribute, y ='porcentaje', hue=groupby,ax=ax)
    plt.xticks(rotation=ticks_rotation)
    plt.title(title)




def kdeplot_per_classes(df, attribute, groupby, title=None, ticks_rotation=0, ax=None):
    """
    Grafica el kdeplot de atributos para cada clase.
    
    :param df: dataframe  con la data que  sera graficada en el kdeplot.
    :param attribute: nombre del atributo a ser graficado en el kdeplot.
    :param groupby: nombre del atributo con la clase predictora.
    :param title: titulo del grafico.
    :param ticks_rotation:  rotacion sobre  x-ticks (etiquetas).
    :param ax: objetos de eje de matplotlib considerado para la grafica.
    """
    for x in df[groupby].unique():
        sns.kdeplot(df[df[groupby] == x][attribute], label=x, shade=True, shade_lowest=False, ax=ax)
    plt.title(title)
    plt.xticks(rotation=ticks_rotation)
    plt.legend()


def boxplot_per_classes(df, attribute, groupby, title=None, ticks_rotation=0, ax=None):
    """
    Grafica el boxplot de atributos para cada clase.
    
    :param df: dataframe  con la data que  sera graficada en el boxplot.
    :param attribute: nombre del atributo a ser graficado en el boxplot.
    :param groupby: nombre del atributo con la clase predictora.
    :param title: titulo del grafico.
    :param ticks_rotation:  rotacion sobre  x-ticks (etiquetas).
    :param ax: objetos de eje de matplotlib considerado para la grafica.
    """
    sns.boxplot(x=groupby, y=attribute, data=df, ax=ax)
    plt.title(title)
    plt.xticks(rotation=ticks_rotation)

######______________________________________________________________________________________________######

###  funciones para hacer el reprocesado de la informacion

class AgrupadorDeClases(TransformerMixin):
    """
    Class que hereda el metodo fit_tranform de TransformerMixin para generar agrupaciones categoricas sobre 
    un campo especificado
    
    :param cols: columnas a tranformar ( se ingresan en una lista)
    :param n_grupos: cantidad de grupos a formar.
    """
    def __init__(self, cols, n_grupos=10):
        self.cols = cols
        self.n_grupos = n_grupos
        
    def fit(self, X, y=None):
        # contar la frecuencia de cada clase en cada columna
        self.clase_counts = {}
        for col in self.cols:
            self.clase_counts[col] = X[col].value_counts()
        # determinar el umbral para agrupar las clases
        umbral = {}
        for col in self.cols:
            umbral[col] = self.clase_counts[col].nsmallest(self.n_grupos).iloc[-1]
        # obtener las clases a agrupar en "otro"
        self.clases_a_agrupar = {}
        for col in self.cols:
            self.clases_a_agrupar[col] = set(self.clase_counts[col][self.clase_counts[col] < umbral[col]].index)
        return self
    
    def transform(self, X, y=None):
        # agrupar las clases con menor frecuencia en una sola clase llamada "otro"
        for col in self.cols:
            X.loc[X[col].isin(self.clases_a_agrupar[col]), col] = 'otro'
        return X


def eliminar_outliers1(df, col_list, n_std):
    """
    función que identifica y elimina outlier sobre un conjunto de datos 
    
    :param df: Dataframe de pandas con la información a evaluar
    :param col_list: columnas a tranformar ( se ingresan en una lista)
    :param n_std: desviasion estandar permitida ( umbral) sobre los datos atipicos
    """
    # calcular la media y la desviación estándar de las columnas especificadas
    col_mean = df[col_list].mean()
    col_std = df[col_list].std()
    
    # identificar los outliers en cada columna
    outliers = np.abs(df[col_list] - col_mean) > n_std * col_std
    
    # reemplazar los outliers con NaN
    df[col_list] = df[col_list].where(~outliers, np.nan)
    
    # eliminar las filas que contienen NaN
    df = df.dropna()
    
    return df


def eliminar_outliers(dataframe, columnas):
    df = dataframe.copy()
    for columna in columnas:
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        df = df[(df[columna] > limite_inferior) & (df[columna] < limite_superior)]
    return df
    



class Homologar(TransformerMixin):
    """
    Class que hereda el metodo fit_tranform de TransformerMixin para generar homologacion de categoricas sobre 
    un campo especificado y una diccionario de mapeo
    
    :param col_name: columnas a tranformar ( se ingresan en una lista)
    :param dictionary: diccionario con los datos  a homologar.
    """
    
    def __init__(self, col_name, dictionary):
        self.col_name = col_name
        self.dictionary = dictionary
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X[self.col_name] = X[self.col_name].map(self.dictionary).fillna(X[self.col_name])
        return X


class MissingValueImputer(TransformerMixin):
    """
    Class que hereda el metodo fit_tranform de TransformerMixin para generar imputacion de datos faltnates  en 
    variables numericas considerando la media y en  variables categoricas  utilizando el valor mas frecuente,
    ademas se considera que la columna no supere un 15% en datos faltantes
    
    :param numerical_columns: lista de columnas numericas a tranformar
    :param numerical_columns: lista de columnas numericas a tranformar
    """
    def __init__(self, numerical_columns, categorical_columns):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Imputa valores perdidos en columnas numéricas
        for column in self.numerical_columns:
            if X[column].isnull().sum() > 0.15*len(X):
                X[column].fillna(X[column].mean(), inplace=True)

        # Imputa valores perdidos en columnas categóricas
        for column in self.categorical_columns:
            if X[column].isnull().sum() > 0.15*len(X):
                X[column].fillna(X[column].value_counts().index[0], inplace=True)

        return X
    


#################################### funciones para despliegue y  ###########################################

def update_model(model: Pipeline) -> None:
    dump(model, 'model/model.pkl')


def save_simple_metrics_report(train_score: float, test_score: float, validation_score: float, model: Pipeline) -> None:
    with open('report.txt', 'w') as report_file:

        report_file.write('# Model Pipeline Description')

        for key, value in model.named_steps.items():
            report_file.write(f'### {key}:{value.__repr__()}'+'\n')

        report_file.write('### Train Score: {train_score}'+'\n')
        report_file.write('### Test Score: {test_score}'+'\n')
        report_file.write('### Validation Score: {validation_score}'+'\n')

def get_model_performance_test_set(y_real: pd.Series, y_pred: pd.Series) ->None:
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    sns.regplot(x=y_pred, y=y_real, ax = ax)
    ax.set_xlabel('Predicted worldwide gross')
    ax.set_ylabel('Real worldwide gross')
    ax.set_title('Behavior of model prediction')
    fig.savefig('prediction_behavior.png')