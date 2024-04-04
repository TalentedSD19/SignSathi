from fastapi import FastAPI
import joblib
from pydantic import BaseModel

class Sensor(BaseModel):
    values: list

app = FastAPI()

model = joblib.load('ICCPDM_3.pkl')

@app.post("/predict")
async def predict(sensor:Sensor):
    output = model.predict([sensor.values])
    return output[0].item()
