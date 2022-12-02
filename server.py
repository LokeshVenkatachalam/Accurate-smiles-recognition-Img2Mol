from cddd.inference import InferenceModel
import fastapi
from pydantic import BaseModel
from typing import List
import numpy as np

app = fastapi.FastAPI()

model = InferenceModel()

class CDDD2Mol(BaseModel):
    cddd: List[float]
    seq: str

@app.post("/predict")
def predict(item: CDDD2Mol):
    embedding = np.array(item.cddd)
    item.seq = model.emb_to_seq(embedding)
    return item
