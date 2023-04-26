import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

cols = ['i', 
        'pub_key_len',
        'block_count', 
        'population',
        'block_num',
        'convergence_time',
        'best_fitness',
        'average_fitness',
        'solution_found',
        'num_solutions_found',
        ]

def isolateDF(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    # stratify df by default values, EXCEPT for our target feature
    if feature != 'pub_key_len':
        df = df.loc[df['pub_key_len'] == 8]  

    if feature != 'block_count':
        df = df.loc[df['block_count'] == 2]  

    if feature != 'population':
        df = df.loc[df['population'] == 64]  

    return df

def getFeatureValues(feature: str) -> list[float]:
    # get list of target feature variable values
    match feature:
        case 'pub_key_len':
            feature_values = [4,8,12,16]
        case 'block_count':
            feature_values = [2,3,4,5]
        case 'pop':
            feature_values = [32,64,128,256]

    return feature_values

def createHeatmap(df: pd.DataFrame) -> None:

    corrcoef = []

    x_labels = []
    y_labels = []

    for pkl in [16,12,8,4]:
        row = []
        for bc in [2,3,4,5]:
            temp = temp.loc[temp['pub_key_len'] == pkl]  
            temp = temp.loc[temp['block_count'] == bc]  
            
            heat = temp.loc[:, 'average_fitness'].mean()
            row.append( round(heat, 2) )
            x_labels.append(str(bc))

        corrcoef.append(row)
        y_labels.append(str(pkl))
    
    fig, ax = plt.subplots(1,1)        
    c = ax.pcolor(corrcoef)
    fig.colorbar(c, ax=ax)

    # loop over data, create annotations for heatmap
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j+0.5, i+0.5, corrcoef[i][j], ha="center", va="center", color='w')

    plt.xticks(np.arange(0.5, len(x_labels), 1), x_labels, fontsize=8, rotation = 90)
    plt.yticks(np.arange(0.5, len(y_labels), 1), y_labels, fontsize=8)

    plt.title("How Public Key Length and Cipher Block Count Affect Average Fitness")
    plt.show()

def plotColVersusColWhere(df: pd.DataFrame, col1: str, col2: str, where: str, val: float) -> None:
    # evaluate where 'where' is equal to 'val'
    # eg where df['population] == 64
    df = df.loc[df[where] == val]  
    
    # get mean and std col2 based off of col1 values
    df = df.groupby(col1, as_index=False)[col2].agg([np.mean, np.std])
    df = df.reset_index()
    df.columns = [col1, 'mean', 'std']

    # plot with error
    plt.plot(df[col1], df['mean'])
    plt.fill_between(df[col1], df['mean']-df['std'], df['mean']+df['std'], alpha=0.4, label=f"{where}={val}")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'{col2} versus convergence time for {where}')
    plt.legend()

def isolateAndPlot(df: pd.DataFrame, feature: str) -> None:
    # get the unique values for a variable and set the rest to defaults
    isolated_df = isolateDF(df, feature)
    feature_values = getFeatureValues(feature)

    # for yaxis in ['average_fitness']:
    for feature_value in feature_values:
        plotColVersusColWhere(isolated_df, 'convergence_time', 'average_fitness', feature, feature_value)

    plt.show()
 

if __name__ == '__main__':
    df = pd.read_csv('cryptanalysis_csvs/0-compiled.csv')

    feature_list = ['pub_key_len', 'block_count', 'population']

    if True:
        for feature in feature_list:
            isolateAndPlot(df, feature)

    if True:
        createHeatmap(df)