import os

import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

from biobert import run_ner as bert_ner
import pipeline_config as cfg


THIS_PATH = os.path.dirname(os.path.realpath(__file__))

ner_processor = bert_ner.NerProcessor()

LABEL_KEYS = ner_processor.get_labels()

_FEATURE_KEYS = ["input_ids", "input_mask", "segment_ids", "label_ids"]

model_dir = os.path.join(THIS_PATH, 'models')



def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs."""
    outputs = {}
    for key in _FEATURE_KEYS:
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


def _build_estimator(
        bert_config,
        run_config,
        init_checkpoint,
        num_train_steps,
        num_warmup_steps,
        train_batch_size,
        eval_batch_size,
        predict_batch_size,
        learning_rate=5e-5,
        use_tpu=False,
        use_one_hot_embeddings=False):

    num_labels = len(LABEL_KEYS) + 1

    model_fn = bert_ner.model_fn_builder(
        bert_config=bert_config,
        num_labels=num_labels,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_one_hot_embeddings)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)

    return estimator