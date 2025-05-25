import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from src.inference.inference import load_checkpoint, init_model, inference
from src.postprocess.postprocess import read_db
from src.utils.utils import CFG
from src.dataset.baselineDataset import get_datasets
from src.utils.constant import Optimizers, Models, Augmentations

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

load_dotenv()

CFG["EXPERIMENT_NAME"] = 'identity_policy'
augmentation_name = 'identity_policy'
model_name = 'resnet_50'

checkpoint = load_checkpoint()
model, criterion = init_model(checkpoint, model_name)

Augmentations.validation(augmentation_name)
train_dataset, val_dataset, test_dataset, class_names = get_datasets(Augmentations[augmentation_name.upper()].value)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InferenceInput(BaseModel):
    user_id: int
    content_id: int
    watch_seconds: int
    rating: float
    popularity: float

@app.post("/predict")
async def predict(input_data: InferenceInput):
    try:
        data = np.array([
            input_data.user_id,
            input_data.content_id,
            input_data.watch_seconds,
            input_data.rating,
            input_data.popularity
        ])
            
        image = torch.randn((1, 3, 224, 224))
            
        # print(result, class_names.index(result))
            
        # recommend_df = recommend_to_df(class_names.index(result))
        # write_db(recommend_df, "mlops", "recommend")
        result = inference(model, image, criterion, device, augmentation_name)
        result = class_names.index(result)
        return {"recommended_content_id": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/batch-predict")
async def batch_predict(k: int = 5):
    try:
        recommend = read_db("mlops", "recommend", k=k)
        return {"recommended_content_id": recommend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)