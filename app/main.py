from typing import Union

from fastapi import FastAPI, UploadFile
from PIL import Image
import tensorflow as tf
import numpy as np
import io
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# request affected by server, maybe nginx doing something, test first on local without tesnorflow, normal docker image of python or even without docker
@app.post("/identifyDigit")
async def identifyDigit(imgFile: UploadFile):
    fileContent = await imgFile.read()
    img = Image.open(io.BytesIO(fileContent))
    print (img)
    # return imgFile.filename
    # convert image to array
    imgArr = np.asarray(img, dtype='uint8')

    # imgArr = np.array(imgArr, dtype='uint8')

    print (imgArr)
    # load model
    model = tf.keras.models.load_model('./saved_model1/')

    # get predictions
    print (model.predict(imgArr))