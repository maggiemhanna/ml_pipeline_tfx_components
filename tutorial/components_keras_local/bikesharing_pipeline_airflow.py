import absl
import datetime
from tfx.orchestration.airflow import airflow_dag_runner

#!pip install -r requirements.txt

import os
import pprint
import tempfile
import urllib

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
tf.get_logger().propagate = False
pp = pprint.PrettyPrinter()

import tfx
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input



print('TensorFlow version: {}'.format(tf.__version__))
print('TFX version: {}'.format(tfx.__version__))

# Data file path source
_data_filepath = "gs://bike-sharing-data/"

# This is the path were components results will be stored, as well as the ml metadata store
_pipeline_root = 'tfx-pipeline'

# This is the path where your model will be pushed for serving.
_serving_model_dir = os.path.join(
    _pipeline_root, 'serving_model/bikesharing')

# Set up logging.
absl.logging.set_verbosity(absl.logging.INFO)

# Here, we create an InteractiveContext using default parameters. This will
# use a temporary directory with an ephemeral ML Metadata database instance.
# To use your own pipeline root or database, the optional properties
# `pipeline_root` and `metadata_connection_config` may be passed to
# InteractiveContext. Calls to InteractiveContext are no-ops outside of the
# notebook.
context = InteractiveContext(pipeline_root = _pipeline_root)

example_gen = CsvExampleGen(input=external_input(_data_filepath))
context.run(example_gen)

statistics_gen = StatisticsGen(
    examples=example_gen.outputs['examples'])
context.run(statistics_gen)

schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics'],
    infer_feature_shape=False)
context.run(schema_gen)

example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])
context.run(example_validator)

_bikesharing_constants_module_file = 'bikesharing_constants.py'

_bikesharing_transform_module_file = 'bikesharing_transform.py'

transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_bikesharing_transform_module_file))
context.run(transform)

_bikesharing_trainer_module_file = 'bikesharing_trainer.py'

trainer = Trainer(
    module_file=os.path.abspath(_bikesharing_trainer_module_file),
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(),
    eval_args=trainer_pb2.EvalArgs(num_steps=10))
context.run(trainer)

import bikesharing_constants

KEY_COLUMN = bikesharing_constants.KEY_COLUMN
LABEL_COLUMN = bikesharing_constants.LABEL_COLUMN
NUMERIC_FEATURE_KEYS = bikesharing_constants.NUMERIC_FEATURE_KEYS
CATEGORICAL_FEATURE_KEYS = bikesharing_constants.CATEGORICAL_FEATURE_KEYS

transformed_name = bikesharing_constants.transformed_name

eval_config = tfma.EvalConfig(
    model_specs=[
        # This assumes a serving model with signature 'serving_default'. If
        # using estimator based EvalSavedModel, add signature_name: 'eval' and 
        # remove the label_key.
        tfma.ModelSpec(label_key=LABEL_COLUMN)
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            # The metrics added here are in addition to those saved with the
            # model (assuming either a keras model or EvalSavedModel is used).
            # Any metrics added into the saved model (for example using
            # model.compile(..., metrics=[...]), etc) will be computed
            # automatically.
            # To add validation thresholds for metrics saved with the model,
            # add them keyed by metric name to the thresholds map.
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='MeanSquaredError',
                      threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          upper_bound={'value': 32000}),
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.LOWER_IS_BETTER,
                          absolute={'value':1000})))
            ]
        )
    ],
    slicing_specs=
        # An empty slice spec means the overall slice, i.e. the whole dataset.
        [tfma.SlicingSpec()] + 
        # Data can be sliced along any feature column.
        [tfma.SlicingSpec(feature_keys=[transformed_name(feature_key)]) for feature_key in CATEGORICAL_FEATURE_KEYS]
)

# Use TFMA to compute a evaluation statistics over features of a model and
# validate them against a baseline.

# The model resolver is only required if performing model validation in addition
# to evaluation. In this case we validate against the latest blessed model. If
# no model has been blessed before (as in this case) the evaluator will make our
# candidate the first blessed model.
model_resolver = ResolverNode(
      instance_name='latest_blessed_model_resolver',
      resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing))
context.run(model_resolver)

evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    # Change threshold will be ignored if there is no baseline (first run).
    eval_config=eval_config)
context.run(evaluator)

pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory=_serving_model_dir)))
context.run(pusher)

_runner_type = 'airflow' #@param ["beam", "airflow"]
_pipeline_name = 'bikesharing_pipeline_%s' % _runner_type

#docs_infra: no_execute

_notebook_filepath = os.path.join(os.getcwd(), 'components_keras_local.ipynb')

_data_root = "gs://bike-sharing-data/"
_tfx_root = os.path.join(os.getcwd(), 'tfx_pipeline')
_serving_model_dir = os.path.join(_tfx_root, 'serving_model')
_pipeline_root = os.path.join(_tfx_root, 'pipelines', _pipeline_name)
_metadata_path = os.path.join(_tfx_root, 'metadata', _pipeline_name,
                              'metadata.db')

#docs_infra: no_execute
components = [
    example_gen, statistics_gen, schema_gen, example_validator, transform,
    trainer, evaluator, pusher
]



# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}

absl.logging.set_verbosity(absl.logging.INFO)

tfx_pipeline = pipeline.Pipeline(
    pipeline_name=_pipeline_name,
    pipeline_root=_pipeline_root,
    components=components,
    enable_cache=True,
    metadata_connection_config=(
        metadata.sqlite_metadata_connection_config(_metadata_path)),

    # We use `--direct_num_workers=1` by default to launch 1 Beam worker process
    # during Beam DirectRunner component execution. This mitigates issues with
    # GPU memory usage when many workers are run sharing GPU resources.  Change
    # this to `--direct_num_workers=0` to run one worker per available CPU
    # thread or `--direct_num_workers=$N`, where `$N` is a fixed number of
    # worker processes.
    #
    # TODO(b/142684737): The Beam multi-processing API might change.
    beam_pipeline_args = ['--direct_num_workers=1'],

    additional_pipeline_args={})

# 'DAG' below needs to be kept for Airflow to detect dag.
DAG = airflow_dag_runner.AirflowDagRunner(
    airflow_dag_runner.AirflowPipelineConfig(_airflow_config)).run(
      tfx_pipeline)