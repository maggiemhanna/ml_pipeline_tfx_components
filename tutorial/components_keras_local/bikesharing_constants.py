
# Features, labels, and key columns
NUMERIC_FEATURE_KEYS=["temp", "atemp", "humidity", "windspeed"] 
CATEGORICAL_FEATURE_KEYS=["season", "weather", "daytype"] 
KEY_COLUMN = "datetime"
LABEL_COLUMN = "count"

# Modeling parameters
HIDDEN_UNITS_1 = 16
HIDDEN_UNITS_2 = 16
HIDDEN_UNITS_3 = 16
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 0.005

def transformed_name(key):
    return key 
