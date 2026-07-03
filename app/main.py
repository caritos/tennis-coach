"""FastAPI web app: upload a forehand clip, get back annotated video + coaching feedback."""
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "ok"}
