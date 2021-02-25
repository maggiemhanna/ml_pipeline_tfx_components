## Tutorial

This tutorial will interactively walk through each built-in component of TensorFlow Extended (TFX) on a jupyter notebook.

It covers every step in an end-to-end machine learning pipeline, from data ingestion to pushing a model to serving.

When you're done, the contents of this notebook can be automatically exported as TFX pipeline source code, which you can orchestrate with Apache Airflow and Apache Beam.

Note: This notebook demonstrates the use of native Keras models in TFX pipelines.

## Data

The model uses the bike sharing demand data from kaggle as a use case for implementation: https://www.kaggle.com/c/bike-sharing-demand/data
The raw data from kaggle can be found 
```
doc/bike-sharing-demand
```

For educational purpose, some of the numeric variables (actually categorical) in the dataset have been changed to string.
This allow us to better interpret these variables (especially for TFMA slices analysis). The data is also copied to a gcp bucket.

You can check how data have been prepared in `doc/0_preparing_data`.

The train, val and test splits of the prepared data can be found at `doc/bike-sharing-data`, as well as the gs bucket `gs://bike-sharing-data`.

## Tutorial

In a production deployment of TFX, you will use an orchestrator such as Apache Airflow, Kubeflow Pipelines, or Apache Beam to orchestrate a pre-defined pipeline graph of TFX components. In an interactive notebook, the notebook itself is the orchestrator, running each TFX component as you execute the notebook cells.

In a production deployment of TFX, you will access metadata through the ML Metadata (MLMD) API.  MLMD stores metadata properties in a database such as MySQL or SQLite, and stores the metadata payloads in a persistent store such as on your filesystem.  In an interactive notebook, both properties and payloads are stored in an ephemeral SQLite database in the `/tmp` directory on the Jupyter notebook or Colab server.


There are two main notebooks in this tutorial:

* `tutorial/components_keras_local/components_keras_local.ipynb`:  ML Metadata payloads and properties are stored in an ephemeral SQLite database in a local directory.

* `tutorial/components_keras_local/components_keras_gs_cloud_sql.ipynb`:  ML Metadata payloads and properties are stored in an Cloud Sql Database and Google Storage Bucket.