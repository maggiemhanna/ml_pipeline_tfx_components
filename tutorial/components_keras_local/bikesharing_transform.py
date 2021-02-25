
import tensorflow as tf
import tensorflow_transform as tft

import bikesharing_constants

NUMERIC_FEATURE_KEYS = bikesharing_constants.NUMERIC_FEATURE_KEYS
CATEGORICAL_FEATURE_KEYS = bikesharing_constants.CATEGORICAL_FEATURE_KEYS
KEY_COLUMN = bikesharing_constants.KEY_COLUMN
LABEL_COLUMN = bikesharing_constants.LABEL_COLUMN

transformed_name = bikesharing_constants.transformed_name

# A function to scale numerical features and label encode categorical features
def preprocessing_fn(inputs):
      
    outputs = {}
    
    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(_fill_in_missing(inputs[key]))
    
    for key in CATEGORICAL_FEATURE_KEYS:    
        outputs[transformed_name(key)] = _fill_in_missing(inputs[key])
        tft.vocabulary(inputs[key], vocab_filename=key)    

    outputs[transformed_name(LABEL_COLUMN)] = _fill_in_missing(inputs[LABEL_COLUMN])
    
    return outputs

def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
    Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)
