import os
import uuid

import uvicorn
from fastapi import FastAPI, File, UploadFile

from models import evaluate_model

app = FastAPI()

DATA_DIR = "data"
MODEL_NAME = "mobilenet_v3_large"

# Create the data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)



@app.post("/get_car_angle")
def chat(file: UploadFile = File(...)):
    """ API to get the car angle and confidence """

    # Get a random image name
    image_name = f"{DATA_DIR}/{uuid.uuid4()}.jpg"
    with open(image_name, "wb") as f:
        f.write(file.file.read())

    # Evaluate the model
    # Using the mobilenet_v3_large model, as it faster than the others
    pred, conf = evaluate_model(MODEL_NAME, image_name)
    return {"pred_class": pred, "confidence": conf}


if __name__ == "__main__":
    # Run the API
    uvicorn.run(app, host="0.0.0.0", port=8000)
