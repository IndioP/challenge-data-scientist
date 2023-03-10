"""Main module."""
import uvicorn
from fastapi import FastAPI
from api.routers import router

from Registro import Registro

app = FastAPI(title='Monitoramento de modelos', version="1.0.0")

@app.get("/")
def read_root():
    """Hello World message."""
    return {"Hello World": "from FastAPI"}

@app.post("/")
def post_data(registro: Registro):
    return {"name":registro.VAR2,"idade":registro.IDADE,"TARGET":registro.TARGET}

app.include_router(router, prefix="/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
