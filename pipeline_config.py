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

# Logging overrides
logger_overrides = {
    'log_root': log_root,
    'log_level': logging.INFO
}
