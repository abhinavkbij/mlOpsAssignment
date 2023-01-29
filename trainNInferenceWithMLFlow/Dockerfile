FROM tensorflow/tensorflow:2.9.3

WORKDIR /code

RUN pip install tensorflow-datasets mlflow[extras]

COPY ./train.py /code/
COPY ./inference.py /code/

CMD ["python", "train.py"]
# CMD ["python", "inference.py"]
