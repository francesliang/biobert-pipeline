import os
import datetime
import logging

THIS_PATH = os.path.dirname(os.path.realpath(__file__))


# Pipeline
pipeline_name = "biobert-pipeline"
pipeline_root = os.path.dirname(os.path.realpath(__file__))
airflow_root = os.path.join(os.environ['HOME'], 'airflow')
airflow_data_root = os.path.join(airflow_root, "data", pipeline_name)
airflow_pipeline_root = os.path.join(airflow_root, pipeline_name)
metadata_db_root = os.path.join(airflow_pipeline_root, 'metadata', 'metadata.db')
log_root = os.path.join(airflow_pipeline_root, 'logs')

module_file = os.path.join(airflow_root, "dags", pipeline_name, "pipeline_utils.py")
tfrecord_dir = os.path.join(airflow_data_root, "tfrecords")
serving_model_dir = os.path.join(airflow_pipeline_root, "models")

airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}
logger_overrides = {
    'log_root': log_root,
    'log_level': logging.INFO
}   # Logging overrides


# Pipeline Utils
use_tpu = False
bert_config_file = os.path.join(THIS_PATH, "biobert_v1.0_pubmed_pmc/bert_config.json")
max_seq_length = 128
output_dir = "outputs"
save_checkpoints_steps = 1000
iterations_per_loop = 1000
num_tpu_cores = 8   # only used if `use_tpu` is True.
train_batch_size = 32
eval_batch_size = 8
predict_batch_size = 8
learning_rate = 5e-5
init_checkpoint = os.path.join(THIS_PATH, "biobert_v1.0_pubmed_pmc/biobert_model.ckpt")
warmup_proportion = 0.1

# Data
data_dir = ""
vocab_file = os.path.join(THIS_PATH, "biobert_v1.0_pubmed_pmc/vocab.txt")
