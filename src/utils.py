from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
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