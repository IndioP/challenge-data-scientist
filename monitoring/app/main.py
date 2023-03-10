"""Main module."""
import uvicorn
from fastapi import FastAPI
from api.routers import router
from typing import List

import pandas as pd

import pickle
import numpy as np

from scipy.stats import kstest, mannwhitneyu

from Registro import Registro
from sklearn.metrics import roc_auc_score

app = FastAPI(title='Monitoramento de modelos', version="1.0.0")

def calculate_volumetry(registros):
    #Filtrando a informação relevante(só as datas)
    full_dates = [registro.REF_DATE for registro in registros]
    #Pegando o nome dos meses das datas de referencia
    dates = [pd.to_datetime(date).strftime('%B') for date in full_dates]
    #Contando a frequencia de cada mes
    volumetria = pd.Series(dates).value_counts()
    #retornando um dicionario com a frequencia para cada mes
    return volumetria.to_dict()

def calculate_ROC():
    #load model and dataset
    model = load_model()
    df = load_dataset()
    #replace null values
    df = df.fillna(value=np.nan)
    y_true = df["TARGET"]
    #predict labels using the model
    pred = model.predict_proba(df.drop(["REF_DATE","TARGET"],axis=1))[:,1]
    #Calculate the area under ROC curve
    return roc_auc_score(y_true,pred)

def load_model():
    with open("../model.pkl","rb") as f:
        return pickle.load(f)
    
def load_dataset(file_path = '../batch_records.json'):
    return pd.read_json(file_path)

def load_compressed_dataset(file_path):
    return pd.read_csv(file_path,compression='gzip')

@app.get("/")
def read_root():
    """Hello World message."""
    return {"Hello World": "from FastAPI"}

@app.post("/v1/")
def post_data(registros: List[Registro]):
    volumetria = calculate_volumetry(registros)
    ROC_AUC = calculate_ROC()
    return {"volumetria":volumetria,"ROC-AUC":ROC_AUC}
    
@app.post("/v2/")
def post_data(file_path: str, test:str = 'KS'):
    #load the model
    model = load_model()

    #load the datasets
    df_test = load_compressed_dataset("../../datasets/credit_01/test.gz")
    df_input = load_compressed_dataset(file_path)
    
    #filter Null values
    df_test.fillna(value=np.nan,inplace=True)
    df_input.fillna(value=np.nan,inplace=True)
    
    #predict both datasets
    pred_test = model.predict_proba(df_test.drop(["REF_DATE","TARGET"],axis=1))[:,1]
    #oot doesn't have the target column
    try:
        df_input = df_input.drop(["REF_DATE","TARGET"],axis=1)
    except:
        df_input = df_input.drop(["REF_DATE"],axis=1)
        df_input = df_input.replace('MUITO PROXIMO',np.NAN)  #cleaning the unexpected value
    pred_input = model.predict_proba(df_input)[:,1]
    if test == 'MAN':
        # é um teste estatistico mais conhecido e equivalente ao KS
        #ambos são nonparametrics e podem ser usados para comparar grupos não pareados
        s,p = mannwhitneyu(pred_test, pred_input)
        return {"estatistica de mannwithneyu":s,"p":p}
    else:
        s,p = kstest(pred_test, pred_input)
        return {"estatistica de Kolmogorov-Smirnov":s,"p":p}
    

app.include_router(router, prefix="/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
