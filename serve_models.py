import os, sys
import collections

from google.protobuf import text_format

from tfx.utils.dsl_utils import tfrecord_input
import tensorflow as tf
from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_metadata.proto.v0 import schema_pb2

from tensorflow.python.lib.io import file_io

THIS_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(THIS_PATH)

from biobert import run_ner as bert_ner
from biobert import tokenization


ner_processor = bert_ner.NerProcessor()


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


def construct_inputs(input_dir, vocab_file, output_dir, proto_coder):
    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=False)
    label_map = bert_ner.get_label_map(ner_processor.get_labels(), output_dir)
    examples = ner_processor.get_test_examples(input_dir)
    input_data = []
    mode = "test"
    max_seq_length = 128
    for ind, example in enumerate(examples):
        feature = bert_ner.convert_single_example(ind, example, label_map, max_seq_length, tokenizer, mode)
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        #features["label_mask"] = create_int_feature(feature.label_mask)
        raw_features = collections.OrderedDict()
        raw_features["input_ids"] = feature.input_ids
        raw_features["input_mask"] = feature.input_mask
        raw_features["segment_ids"] = feature.segment_ids
        raw_features["label_ids"] = feature.label_ids

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        print('tf_example type', type(tf_example))
        #input_data.append(tf_example.SerializeToString())
        input_data.append(proto_coder.encode(raw_features))

    return input_data


def do_inference(model_dir, input_dir, schema_file, vocab_file, output_dir):
    saved_model = tf.saved_model.load_v2(model_dir, tags=['serve'])
    infer = saved_model.signatures['serving_default']
    print(infer.__dict__)

    schema = _read_schema(schema_file)
    proto_coder = _make_proto_coder(schema)

    input_data = construct_inputs(input_dir, vocab_file, output_dir, proto_coder)
    #dataset = tf.data.TFRecordDataset(input_tfrecord_file)
    #parsed_dataset = dataset.map(read_tfrecord)

    prediction = infer(input_data)
    return prediction


if __name__ == "__main__":
    #model_dir = sys.argv[1]
    #input_tfrecord_file = sys.argv[2]
    model_dir = "/Users/xinliang/airflow/biobert-pipeline/Trainer/output/40/serving_model_dir/export/biobert-pipeline/1571268068/"
    #input_tfrecord_file = "/Users/xinliang/Projects/drug-name-recogniser/biobert-pipeline/outputs/test.tfrecords"
    input_dir = "/Users/xinliang/Projects/drug-name-recogniser/NERdata/BC4CHEMD/"
    schema_file = "/Users/xinliang/airflow/biobert-pipeline/SchemaGen/output/3/schema.pbtxt"
    vocab_file = "/Users/xinliang/Projects/drug-name-recogniser/biobert-pipeline/biobert_v1.0_pubmed_pmc/vocab.txt"
    output_dir = "/Users/xinliang/Projects/drug-name-recogniser/biobert-pipeline/outputs"
    res = do_inference(model_dir, input_dir, schema_file, vocab_file, output_dir)
    print(res)
