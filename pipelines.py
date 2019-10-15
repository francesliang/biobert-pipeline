import sys
import os
import datetime
import logging

from tfx import components
from tfx.utils.dsl_utils import tfrecord_input
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.transform.component import Transform
from tfx.components.trainer.component import Trainer
from tfx.components.evaluator.component import Evaluator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher

from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner

from tfx.proto import trainer_pb2
from tfx.proto import pusher_pb2

from pipeline_config import *


def create_pipelines():

    examples = tfrecord_input(tfrecord_dir)
    example_gen = ImportExampleGen(input_base=examples)
    print('example-gen', example_gen.outputs.examples)

    statistics_gen = StatisticsGen(
        input_data = example_gen.outputs.examples)
    print('statistics-gen', statistics_gen.outputs.output)

    infer_schema = SchemaGen(
        stats=statistics_gen.outputs.output,
        infer_feature_shape=True)
    print('schema-gen', infer_schema.outputs.output)

    validate_stats = ExampleValidator(
        stats=statistics_gen.outputs.output,
        schema=infer_schema.outputs.output)
    print('example-validator', validate_stats.outputs.output)

    transform = Transform(
        input_data=example_gen.outputs.examples,
        schema=infer_schema.outputs.output,
        module_file=module_file)
    print('transform', transform.outputs.transformed_examples)

    trainer = Trainer(
        module_file=module_file,
        transformed_examples=transform.outputs.transformed_examples,
        schema=infer_schema.outputs.output,
        transform_output=transform.outputs.transform_output,
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50))
    print('trainer', trainer.outputs.output)

    model_analyzer = Evaluator(
        examples=example_gen.outputs.examples,
        model_exports=trainer.outputs.output
        )
    print('model_analyzer', model_analyzer)

    model_validator = ModelValidator(
        examples=example_gen.outputs.examples,
        model=trainer.outputs.output)
    print('model_validator', model_validator)

    pusher = Pusher(
        model_export=trainer.outputs.output,
        model_blessing=model_validator.outputs.blessing,
        push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir))
    )
    print('pusher', pusher)


    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=airflow_pipeline_root,
        components=[
            example_gen, statistics_gen, infer_schema, validate_stats, transform,
            trainer, model_analyzer, model_validator, pusher
        ],
        enable_cache=True,
        metadata_db_root=metadata_db_root,
        additional_pipeline_args={'logger_args': logger_overrides}
    )


DAG = AirflowDagRunner(airflow_config).run(
    create_pipelines())

