import random
import datetime
import fire

from google.cloud import aiplatform


def main(
    project: str,
    location: str,
    experiment_name: str,
    experiment_description: str,
    tensorboard_uri: str,
):
    # Init
    aiplatform.init(
        experiment=experiment_name,
        experiment_description=experiment_description,
        project=project,
        location=location,
    )
    run_name = f"sample-run-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    tensorboard = aiplatform.Tensorboard(tensorboard_uri)

    for i in range(10):
        # Execute experiment
        aiplatform.start_run(run=f"{run_name}-{i}", tensorboard=tensorboard)
        # Store parameters of experiment
        parameters = {"epoch": random.randint(1, 64), "batch_size": random.randint(1, 64), "lr": 0.1 * i, "iter": i}
        aiplatform.log_params(parameters)
        # Store metrics of experiment results
        aiplatform.log_metrics(
            {
                "accuracy": random.uniform(0, 1),
                "recall": random.uniform(0, 1),
                "precision": random.uniform(0, 1),
            }
        )
        # Plot classification metrics
        aiplatform.log_classification_metrics(
            labels=["Positive", "Negative"],
            matrix=[
                [random.randint(0, 100), random.randint(0, 100)],
                [random.randint(0, 100), random.randint(0, 100)],
            ],
            fpr=[random.uniform(0.1, 0.3) for _ in range(101)],
            tpr=[random.uniform(0.6, 0.9) for _ in range(101)],
            threshold=[i / 100 for i in range(101)],
            display_name="Sample of classification metrics",
        )
        aiplatform.end_run()


if __name__ == "__main__":
    fire.Fire(main)
