from fastapi import FastAPI, UploadFile
from PIL import Image
import tensorflow as tf
import numpy as np
import io


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/identifyDigit")
async def identifyDigit(imgFile: UploadFile):
    fileContent = await imgFile.read()

    img = Image.open(io.BytesIO(fileContent))

    # convert image to array
    imgArr = np.asarray(img, dtype='uint8')

    # add channel
    imgArr = np.expand_dims(imgArr, axis=2)
    # add batch dim
    imgArr = np.expand_dims(imgArr, axis=0)

    # load model
    model = tf.keras.models.load_model('/code/app/saved_model2/')

    # get predictions
    predictions = model.predict(imgArr)
    
    return {"digit": int(np.argmax(predictions[0]))}
