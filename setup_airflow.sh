
PROJECT_NAME="biobert-pipeline"
AIRFLOW_DAGS_PATH=~/airflow/dags/$PROJECT_NAME
AIRFLOW_DATA_PATH=~/airflow/data/$PROJECT_NAME
AIRFLOW_PROJ_PATH=~/airflow/$PROJECT_NAME

# Create directories in airflow home directory
mkdir ~/airflow/dags
mkdir ~/airflow/data

# Create project directories in airflow home directory
echo "Create project directories in airflow home directory"
mkdir $AIRFLOW_DAGS_PATH
mkdir $AIRFLOW_DATA_PATH
mkdir $AIRFLOW_DATA_PATH/tfrecords
mkdir $AIRFLOW_PROJ_PATH
mkdir $AIRFLOW_PROJ_PATH/logs
mkdir $AIRFLOW_PROJ_PATH/metadata

# Copy files to airflow home directory
echo "Copy project files to airflow home directory"
cp -r pipelines.py pipeline_utils.py pipeline_config.py biobert biobert_v1.0_pubmed_pmc $AIRFLOW_DAGS_PATH
cp -r outputs/*.tfrecords $AIRFLOW_DATA_PATH

# Copy airflow config file to airflow home directory
echo "Copy airflow.cfg to airflow home directory"
cp airflow.cfg ~/airflow

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES


# Refresh Airflow to pick up new config
echo "Refresh Airflow to pick up new config"
airflow resetdb --yes
airflow initdb


