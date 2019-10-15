import os
import sys

this_path = os.path.dirname(os.path.realpath(__file__))
biobert_path = os.path.join(this_path, 'biobert')
sys.path.append(biobert_path)

from biobert import run_ner as bert_ner
from biobert import tokenization

import pipeline_config as cfg


ner_processor = bert_ner.NerProcessor()


def to_tfrecords(
        data_dir,
        output_dir,
        vocab_file,
        do_lower_case=False,
        max_seq_length=128,
        mode='test'):
    tfrecord_file = os.path.join(output_dir, mode+'.tfrecords')
    if mode == 'train':
        examples = ner_processor.get_train_examples(data_dir)
    else:
        examples = ner_processor.get_test_examples(data_dir)
    label_list = ner_processor.get_labels()
    label_map = bert_ner.get_label_map(label_list, output_dir)
    tokeniser = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    bert_ner.filed_based_convert_examples_to_features(
        examples, label_map, max_seq_length, tokeniser, tfrecord_file)


if __name__ == "__main__":
    print("Convert raw data to tfrecords.")
    to_tfrecords(cfg.data_dir, cfg.output_dir, cfg.vocab_file)

