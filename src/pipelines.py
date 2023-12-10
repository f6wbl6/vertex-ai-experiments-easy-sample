import os
import tempfile
import fire

from google.cloud import aiplatform
from kfp.v2 import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import Input, Metrics, Output, component, Model, ClassificationMetrics


@component(base_image="python:3.11.5-slim-bullseye")
def train(
    epoch: int,
    batch_size: int,
    description: str,
    trained_model: Output[Model],
    summary_metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
) -> None:
    import json
    import random
    from pathlib import Path

    print(f"epoch: {epoch}, batch_size: {batch_size}, description: {description}")

    # Store trained model
    Path(trained_model.path).parent.mkdir(parents=True, exist_ok=True)
    with open(trained_model.path, "w") as f:
        json.dump({"dummy_data": "for_test"}, f)

    # Store metrics
    summary_metrics.log_metric("accuracy", 0.9)
    summary_metrics.log_metric("recall", 0.8)
    summary_metrics.log_metric("precision", 0.7)
    summary_metrics.log_metric("description", description)

    # Plot ROC curve
    classification_metrics.log_roc_curve(
        fpr=[random.uniform(0.1, 0.3) for _ in range(101)],
        tpr=[random.uniform(0.6, 0.9) for _ in range(101)],
        threshold=[i / 100 for i in range(101)],
    )

    # Plot confusion matrix
    matrix = [
        [random.randint(0, 100), random.randint(0, 100)],
        [random.randint(0, 100), random.randint(0, 100)],
    ]
    classification_metrics.log_confusion_matrix(
        categories=["Positive", "Negative"], matrix=matrix
    )


@component(base_image="python:3.11.5-slim-bullseye")
def predict(test_data: str, trained_model: Input[Model]) -> None:
    import json

    print(test_data)
    with open(trained_model.path) as f:
        print(json.load(f))


@dsl.pipeline(name="sample-pipeline")
def pipeline(epoch: int, batch_size: int, train_description: str):
    train_task = train(
        epoch=epoch, batch_size=batch_size, description=train_description
    )
    predict(
        test_data="Path of test data", trained_model=train_task.outputs["trained_model"]
    )


def main(
    project: str,
    location: str,
    pipeline_root: str,
    display_name: str,
    enable_cache: bool,
    parameters: dict,
    experiment_name: str,
):
    """For test with Vertex AI Pipelines"""
    aiplatform.init(
        project=project,
        location=location,
    )

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, "pipeline.json")
        compiler.Compiler().compile(
            pipeline_func=pipeline, package_path=path, type_check=False
        )

        job = aiplatform.PipelineJob(
            display_name=display_name,
            enable_caching=enable_cache,
            template_path=path,
            parameter_values=parameters,
            pipeline_root=pipeline_root,
        )
        job.submit(
            experiment=experiment_name,
        )


if __name__ == "__main__":
    fire.Fire(main)
