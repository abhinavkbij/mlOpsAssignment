from typing import Union

from fastapi import FastAPI, File
from PIL import Image
import tensorflow as tf
import numpy as np

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/identifyDigit")
def identifyDigit(img: bytes = File()):
    # convert image to array
    imgArr = Image.frombytes(file)

    imgArr = np.array(imgArr, dtype='uint8')

    print (imgArr)
    # load model
    model = tf.keras.models.load_model('/code/app/saved_model1/')

    # get predictions
    model.predict(imgArr)