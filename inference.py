import tensorflow as tf
import tensorflow_datasets as tfds

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
model = tf.keras.models.load_model('./saved_model/')

# get predictions on test data
print (model.predict(ds_test))