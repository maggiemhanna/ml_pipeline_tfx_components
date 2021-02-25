
from typing import List, Text

import os
import absl
import datetime
import tensorflow as tf
import tensorflow_transform as tft

from tfx.components.trainer.executor import TrainerFnArgs

import bikesharing_constants

KEY_COLUMN = bikesharing_constants.KEY_COLUMN
LABEL_COLUMN = bikesharing_constants.LABEL_COLUMN
NUMERIC_FEATURE_KEYS = bikesharing_constants.NUMERIC_FEATURE_KEYS
CATEGORICAL_FEATURE_KEYS = bikesharing_constants.CATEGORICAL_FEATURE_KEYS

HIDDEN_UNITS_1 = bikesharing_constants.HIDDEN_UNITS_1
HIDDEN_UNITS_2 = bikesharing_constants.HIDDEN_UNITS_2
HIDDEN_UNITS_3 = bikesharing_constants.HIDDEN_UNITS_3
BATCH_SIZE = bikesharing_constants.BATCH_SIZE
NUM_EPOCHS = bikesharing_constants.NUM_EPOCHS
LEARNING_RATE = bikesharing_constants.LEARNING_RATE

transformed_name = bikesharing_constants.transformed_name


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')

def get_dataset_size(file_path):
    """Function that fetchs the size of the Tfrecords dataset."""
    size = 1
    file_list = tf.io.gfile.glob(file_path)
    for file in file_list:
        for record in tf.compat.v1.io.tf_record_iterator(file, options=tf.io.TFRecordOptions(
    compression_type='GZIP')):
            size += 1
    return size

def _get_serve_tf_examples_fn(model, label_column, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(label_column)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn

def _input_fn(file_pattern: List[Text],
              label_column,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 16) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

    Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=transformed_name(label_column))

    return dataset



def _build_keras_model(tf_transform_output: tft.TFTransformOutput,
                       hidden_units: List[int] = None,
                       learning_rate: float = 0.01) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying taxi data.

    Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).

    Returns:
    A keras Model.
    """
    numeric_columns = [
      tf.feature_column.numeric_column(transformed_name(key), shape=())
      for key in NUMERIC_FEATURE_KEYS
    ]
    
    categorical_columns = [
      tf.feature_column.categorical_column_with_vocabulary_file(
        transformed_name(key), 
        vocabulary_file=tf_transform_output.vocabulary_file_by_name(
            vocab_filename=key), 
        dtype=tf.dtypes.string,
        default_value=None, 
        num_oov_buckets=0)
      for key in CATEGORICAL_FEATURE_KEYS
    ]
    
    indicator_columns = [
      tf.feature_column.indicator_column(categorical_column)
      for categorical_column in categorical_columns
    ]
        
        
    model = dnn_regressor(
      input_columns=numeric_columns + indicator_columns,
      dnn_hidden_units=hidden_units or [16, 16, 16],
      learning_rate=learning_rate)
    return model


def dnn_regressor(input_columns, dnn_hidden_units, learning_rate):
    """Build a simple keras wide and deep model.

    Args:
    wide_columns: Feature columns wrapped in indicator_column for wide (linear)
      part of the model.
    deep_columns: Feature columns for deep part of the model.
    dnn_hidden_units: [int], the layer sizes of the hidden DNN.

    Returns:
    A Wide and Deep Keras model
    """
    # Following values are hard coded for simplicity in this example,
    # However prefarably they should be passsed in as hparams.

    input_layers = {
      colname: tf.keras.layers.Input(name=transformed_name(colname), shape=(), dtype=tf.float32)
      for colname in NUMERIC_FEATURE_KEYS
    }
    input_layers.update({
      colname: tf.keras.layers.Input(name=transformed_name(colname), shape=(), dtype=tf.string)
      for colname in CATEGORICAL_FEATURE_KEYS
    })

    deep = tf.keras.layers.DenseFeatures(input_columns)(input_layers)
    for numnodes in dnn_hidden_units:
        deep = tf.keras.layers.Dense(numnodes, activation='relu')(deep)

    output = tf.keras.layers.Dense(
      1, activation=None)(deep)

    model = tf.keras.Model(input_layers, output)
    model.compile(
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate),
        loss = "mean_squared_error",
        metrics=[tf.keras.metrics.MeanSquaredError()])
    model.summary(print_fn=absl.logging.info)
    return model


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
    """Train the model based on given args.

    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """

    # Training set size
    TRAIN_SIZE = get_dataset_size(fn_args.train_files)
    NUM_STEPS = TRAIN_SIZE / BATCH_SIZE # number of steps per epoch for which to train model
    
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = _input_fn(fn_args.train_files, LABEL_COLUMN, tf_transform_output, BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, LABEL_COLUMN, tf_transform_output, BATCH_SIZE)

    model = _build_keras_model(
        tf_transform_output,
        hidden_units=[HIDDEN_UNITS_1, HIDDEN_UNITS_2, HIDDEN_UNITS_3],
        learning_rate=LEARNING_RATE)

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir, update_freq='batch')
    
    model.fit(
      train_dataset,
      epochs=NUM_EPOCHS,        
      steps_per_epoch=NUM_STEPS,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

    signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    LABEL_COLUMN,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
