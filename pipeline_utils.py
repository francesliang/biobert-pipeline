import os

import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, InputLayer, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.datasets import mnist


THIS_PATH = os.path.dirname(os.path.realpath(__file__))

LABEL_KEYS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
IM_SHAPE = (28, 28, 1)

_OTHER_FEATURE_KEYS = ["label"]
_FEATURE_KEYS_TO_NORMALISE = ["image_raw"]
_LABEL_KEY = "label"
_INPUT_LAYER = 'input_1'

model_dir = os.path.join(THIS_PATH, 'models')


def _transform_name(key):
    output = key
    if key == "image_raw":
        output = _INPUT_LAYER
    return output


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs."""
    outputs = {}
    max_value = tf.constant(255.0)
    for key in _FEATURE_KEYS_TO_NORMALISE:
        outputs[_transform_name(key)] = tf.divide(inputs[key], max_value)
    for key in _OTHER_FEATURE_KEYS:
        outputs[_transform_name(key)] = inputs[key]
    return outputs


def _get_raw_feature_spec(schema):
    # Tf.Transform considers these features as "raw"
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""

    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(filenames, tf_transform_output, batch_size=2):
    transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

    print('transformed_feature_spec', transformed_feature_spec)

    dataset = tf.data.experimental.make_batched_features_dataset(
      filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

    transformed_features = dataset.make_one_shot_iterator().get_next()

    print('transformed_features', transformed_features)
    # We pop the label because we do not want to use it as a feature while we're
    # training.
    return transformed_features, transformed_features.pop(_LABEL_KEY)


def serving_receiver_fn(tf_transform_output, schema):
    """Build the serving in inputs."""

    raw_feature_spec = _get_raw_feature_spec(schema)
    print("raw_feature_spec", raw_feature_spec)
    raw_feature_spec.pop(_LABEL_KEY)

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    print('serving_input_receiver.features', serving_input_receiver.features)

    transformed_features = tf_transform_output.transform_raw_features(
        serving_input_receiver.features)
    transformed_features.pop(_LABEL_KEY)

    print('transformed_features in serving', transformed_features)

    return tf.estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors)


def eval_input_receiver_fn(tf_transform_output, schema):
    """Build everything needed for the tf-model-analysis to run the model."""
    raw_feature_spec = _get_raw_feature_spec(schema)
    serialized_tf_example = tf.placeholder(
      dtype=tf.string, shape=[None], name='input_example_tensor')

    # Add a parse_example operator to the tensorflow graph, which will parse
    # raw, untransformed, tf examples.
    features = tf.parse_example(serialized_tf_example, raw_feature_spec)

    # Now that we have our raw examples, process them through the tf-transform
    # function computed during the preprocessing step.
    transformed_features = tf_transform_output.transform_raw_features(
        features)

    # The key name MUST be 'examples'.
    receiver_tensors = {'examples': serialized_tf_example}

    #features.update(transformed_features)
    features = {_INPUT_LAYER: transformed_features[_INPUT_LAYER]}

    return tfma.export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=transformed_features[_LABEL_KEY])


def trainer_fn(hparams, schema):
    train_batch_size = 50
    eval_batch_size = 50

    print('Hyperparameters in trainer_fn', hparams.__dict__)

    tf_transform_output = tft.TFTransformOutput(hparams.transform_output)

    train_input_fn = lambda: input_fn(
        hparams.train_files,
        tf_transform_output,
        batch_size=train_batch_size)

    eval_input_fn = lambda: input_fn(
        hparams.eval_files,
        tf_transform_output,
        batch_size=eval_batch_size)

    export_serving_receiver_fn = lambda: serving_receiver_fn(tf_transform_output, schema)
    exporter = tf.estimator.FinalExporter('ml-pipeline', export_serving_receiver_fn)

    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=hparams.train_steps)

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=hparams.eval_steps,
        exporters=[exporter])

    run_config = tf.estimator.RunConfig(model_dir=hparams.serving_model_dir)
    estimator = _build_estimator(config=run_config)

    receiver_fn = lambda: eval_input_receiver_fn(
        tf_transform_output, schema)

    return {
        "estimator": estimator,
        "train_spec": train_spec,
        "eval_spec": eval_spec,
        "eval_input_receiver_fn": receiver_fn
    }


def _build_estimator(config):

    num_classes = len(LABEL_KEYS)

    # Keras example of CNN for MNIST
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(
        input_shape=(IM_SHAPE[0]*IM_SHAPE[1]*IM_SHAPE[2],),
        name='input_1'))
    model.add(tf.keras.layers.Reshape(IM_SHAPE))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
         activation='relu',
         input_shape=IM_SHAPE))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    optimiser = tf.keras.optimizers.Adadelta()

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
        optimizer=optimiser,
        metrics=['accuracy'])

    # Convert a Keras model to tf.Estimator
    estimator = tf.keras.estimator.model_to_estimator(keras_model=model,
        config=config)

    return estimator

