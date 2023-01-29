import tensorflow as tf
import tensorflow_datasets as tfds
import os
import mlflow

tfds.core.utils.gcs_utils._is_gcs_disabled = True
os.environ['NO_GCE_CHECK'] = 'true'

mlflow.set_tracking_uri("http://20.244.8.129:5000")
mlflow.set_experiment("agoodsamaritan")

#from pprint import pprint

#client = MlflowClient()
#for rm in client.search_registered_models():
    #pprint(dict(rm), indent=4)

# load dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


# prepare test data
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# load model
#model = tf.keras.models.load_model('./saved_model/')
model = mlflow.tensorflow.load_model("models:/model1/Staging")

# get predictions on test data
print (model.predict(ds_test))
