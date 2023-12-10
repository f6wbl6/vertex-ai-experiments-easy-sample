# vertex-ai-experiments-easy-sample
This is a sample code for getting started with [Vertex AI Experiments](https://cloud.google.com/vertex-ai/docs/experiments).

# Prerequisites
Create a virtual environment with pyenv and install the libraries for execution with poetry.

```shell
$ pyenv --version
pyenv 2.3.17-10-g920ef145

$ pyenv virtualenv 3.11.3 311-vertex
$ pyenv shell 311-vertex
$ pip install "poetry==1.7.1"
$ poetry install
```

# Execution
```shell
$ cd src
# Execute experiment with ai platform module
$ poetry run python experiments.py \
--project "<YOUR_PROJECT>" \
--location "<YOUR_REGION>" \
--experiment_name "sample-experiment" \
--experiment_description "sample execution from local with aiplatform library" \
--tensorboard_uri "projects/<PROJECT_ID>/locations/<REGION>/tensorboards/<TENSORBOARD_ID>"

# Execute experiment with vertex ai pipelines
$ poetry run python pipelines.py \  
--project "<YOUR_PROJECT>" \
--location asia-northeast1 \
--pipeline_root "<YOUR_REGION>" \
--display_name "sample-pipeline-display-name" \
--enable_cache False \
--parameters "{\"epoch\":20,\"batch_size\":64,\"train_description\":\"increase epoch and batch_size\"}" \
--experiment_name "sample-pipeline-experiment"
```

