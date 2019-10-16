import os
import sys

import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

THIS_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_PATH)

from biobert import modeling
from biobert import run_ner as bert_ner
import pipeline_config as cfg


THIS_PATH = os.path.dirname(os.path.realpath(__file__))

ner_processor = bert_ner.NerProcessor()

LABEL_KEYS = ner_processor.get_labels()

_FEATURE_KEYS = ["input_ids", "input_mask", "segment_ids", "label_ids"]

_LABEL_KEY = "label_ids"

model_dir = os.path.join(THIS_PATH, 'models')



def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs."""
    outputs = {}
    for key in _FEATURE_KEYS:
        outputs[key] = inputs[key]
    return outputs


def _get_raw_feature_spec(schema):
    # Tf.Transform considers these features as "raw"
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""

    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn_builder(filenames, tf_transform_output, batch_size):

    #def input_fn(filenames, tf_transform_output, batch_size=2):
    def input_fn(params):
        batch_size = params['batch_size']
        transformed_feature_spec = (
          tf_transform_output.transformed_feature_spec().copy())

        print('transformed_feature_spec', transformed_feature_spec)

        dataset = tf.data.experimental.make_batched_features_dataset(
          filenames, batch_size, transformed_feature_spec, reader=_gzip_reader_fn)

        transformed_features = dataset.make_one_shot_iterator().get_next()

        print('transformed_features', transformed_features)
        # We pop the label because we do not want to use it as a feature while we're
        # training.
        #return transformed_features, transformed_features.pop(_LABEL_KEY)
        return transformed_features, transformed_features[_LABEL_KEY]
    return input_fn


def serving_receiver_fn(tf_transform_output, schema):
    """Build the serving in inputs."""

    raw_feature_spec = _get_raw_feature_spec(schema)
    print("raw_feature_spec", raw_feature_spec)
    #raw_feature_spec.pop(_LABEL_KEY)

    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    print('serving_input_receiver.features', serving_input_receiver.features)

    transformed_features = tf_transform_output.transform_raw_features(
        serving_input_receiver.features)
    #transformed_features.pop(_LABEL_KEY)

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

    features.update(transformed_features)

    return tfma.export.EvalInputReceiver(
        features=features,
        receiver_tensors=receiver_tensors,
        labels=transformed_features[_LABEL_KEY])


def trainer_fn(hparams, schema):
    print('Hyperparameters in trainer_fn', hparams.__dict__)

    tf_transform_output = tft.TFTransformOutput(hparams.transform_output)

    # input_fn
    train_input_fn = input_fn_builder(
        hparams.train_files,
        tf_transform_output,
        batch_size=cfg.train_batch_size)

    eval_input_fn = input_fn_builder(
        hparams.eval_files,
        tf_transform_output,
        batch_size=cfg.eval_batch_size)

    export_serving_receiver_fn = lambda: serving_receiver_fn(tf_transform_output, schema)
    exporter = tf.estimator.FinalExporter('biobert-pipeline', export_serving_receiver_fn)

    # specs
    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=hparams.train_steps)

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=hparams.eval_steps,
        exporters=[exporter])

    # configs
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    tpu_config = tf.contrib.tpu.TPUConfig(
        iterations_per_loop=cfg.iterations_per_loop,
        num_shards=cfg.num_tpu_cores,
        per_host_input_for_training=is_per_host
    )
    run_config = tf.contrib.tpu.RunConfig(
        cluster=None,
        master=None,
        model_dir=cfg.output_dir,
        save_checkpoints_steps=cfg.save_checkpoints_steps,
        tpu_config=tpu_config
    )
    bert_config = modeling.BertConfig.from_json_file(cfg.bert_config_file)
    num_warmup_steps = int(hparams.train_steps * cfg.warmup_proportion)
    estimator = _build_estimator(
        bert_config=bert_config,
        run_config=run_config,
        init_checkpoint=cfg.init_checkpoint,
        num_train_steps=hparams.train_steps,
        num_warmup_steps=num_warmup_steps,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        predict_batch_size=cfg.predict_batch_size,
        learning_rate=cfg.learning_rate,
        use_tpu=cfg.use_tpu,
        use_one_hot_embeddings=cfg.use_tpu,
        max_seq_length=cfg.max_seq_length)

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
        use_one_hot_embeddings=False,
        max_seq_length=128):

    num_labels = len(LABEL_KEYS) + 1

    model_fn = bert_ner.model_fn_builder(
        bert_config=bert_config,
        num_labels=num_labels,
        init_checkpoint=cfg.init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_one_hot_embeddings,
        max_seq_length=max_seq_length)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)

    return estimator
