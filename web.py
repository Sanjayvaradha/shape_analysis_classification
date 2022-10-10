from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MODEL = tf.keras.models.load_model("new_shape_analysis.h5")

@app.get("/")
async def root(request: Request):
    #return {"Predicting shape analysis. Add '/docs' after 8000 to use the page"}
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ping")
async def ping():
    return "Predicting shape analysis"

def read_file_as_image(data) -> np.ndarray:
    image= Image.open(BytesIO(data)) 
    return image

@app.post("/")
async def predict(
    request: Request,file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image = np.array(image)
    image=cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image=cv2.resize(image ,(256,256))
    image=np.expand_dims(image, axis=0)
    
    predictions = MODEL.predict(image)
    prediction=["good" if i > 0.5 else "defect" for i in predictions]

    # return {
    #     "text": prediction
    #       }

    return templates.TemplateResponse("index.html", {"request": request,'prediction':prediction})
if __name__ == "__main__":
    uvicorn.run(app,debug=True)
