"""Main module."""
import uvicorn
from fastapi import FastAPI
from api.routers import router
from typing import List

import pandas as pd

from Registro import Registro

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

@app.get("/")
def read_root():
    """Hello World message."""
    return {"Hello World": "from FastAPI"}

@app.post("/")
def post_data(registros: List[Registro]):
    volumetria = calculate_volumetry(registros)
    return {"volumetria":volumetria}
    

app.include_router(router, prefix="/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
