"""Main module."""
import uvicorn
from fastapi import FastAPI
from api.routers import router
from typing import List

import pandas as pd

import pickle
import numpy as np

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
    pred = model.predict(df.drop(["REF_DATE","TARGET"],axis=1))
    #Calculate the area under ROC curve
    return roc_auc_score(y_true,pred)

def load_model():
    with open("../model.pkl","rb") as f:
        return pickle.load(f)
def load_dataset():
    return pd.read_json('../batch_records.json')

@app.get("/")
def read_root():
    """Hello World message."""
    return {"Hello World": "from FastAPI"}

@app.post("/v1/")
def post_data(registros: List[Registro]):
    volumetria = calculate_volumetry(registros)
    ROC_AUC = calculate_ROC()
    return {"volumetria":volumetria,"ROC-AUC":ROC_AUC}
    

app.include_router(router, prefix="/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
