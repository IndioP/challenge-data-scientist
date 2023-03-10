"""Main module."""
import uvicorn
from fastapi import FastAPI
from api.routers import router
from typing import List

import numpy as np

from scipy.stats import kstest, mannwhitneyu

from Registro import Registro


from utils import calculate_volumetry, calculate_ROC, load_model, load_dataset, load_compressed_dataset, predict_twin_datasets

app = FastAPI(title='Monitoramento de modelos', version="1.0.0")

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
    pred_test, pred_input = predict_twin_datasets(model,df_test,df_input)
    
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
