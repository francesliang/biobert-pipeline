import os, sys

from google.protobuf import text_format

import tensorflow as tf
from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_metadata.proto.v0 import schema_pb2

from tensorflow.python.lib.io import file_io


def _get_raw_feature_spec(schema):
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def _read_schema(path):
    """Reads a schema from the provided location.
      Args:
          path: The location of the file holding a serialized Schema proto.
      Returns:
          An instance of Schema or None if the input argument is None
    """
    result = schema_pb2.Schema()
    contents = file_io.read_file_to_string(path)
    text_format.Parse(contents, result)
    return result


def _make_proto_coder(schema):
    raw_feature_spec = _get_raw_feature_spec(schema)
    raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
    return tft_coders.ExampleProtoCoder(raw_schema)


def read_tfrecord(serialised_example, seq_length=128):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        #"label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    example = tf.parse_single_example(serialised_example, name_to_features)
    print(type(example))
    print(example)

    return example


def do_inference(model_dir, input_tfrecord_file, schema_file):
    saved_model = tf.saved_model.load_v2(model_dir, tags=['serve'])
    infer = saved_model.signatures['serving_default']

    schema = _read_schema(schema_file)
    proto_coder = _make_proto_coder(schema)

    examples = []
    dataset = tf.data.TFRecordDataset(input_tfrecord_file)
    #parsed_dataset = dataset.map(read_tfrecord)
    for data in dataset:
        #example = proto_coder.encode(data)
        example = data
        examples.append(example)
        prediction = infer(example)
    return prediction


if __name__ == "__main__":
    #model_dir = sys.argv[1]
    #input_tfrecord_file = sys.argv[2]
    model_dir = "/Users/xinliang/airflow/biobert-pipeline/Trainer/output/40/serving_model_dir/export/biobert-pipeline/1571268068/"
    input_tfrecord_file = "/Users/xinliang/Projects/drug-name-recogniser/biobert-pipeline/outputs/test.tfrecords"
    schema_file = "/Users/xinliang/airflow/biobert-pipeline/SchemaGen/output/3/schema.pbtxt"
    res = do_inference(model_dir, input_tfrecord_file, schema_file)
    print(res)
