import pandas as pd
import pickle
import numpy as np

from sklearn.metrics import roc_auc_score

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

def predict_twin_datasets(model,df_test,df_input):
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
    return pred_test,pred_input