import mlflow
import os
import shutil


def clean_phantom_folder():
    """
    At the start of the MLproject run, Mlflow assigns a `run_id`:
    ```
        INFO mlflow.projects.backend.local: 
        === Running command 'source /anaconda3/bin/../etc/profile.d/conda.sh && 
        conda activate mlflow-4736797b8261ec1b3ab764c5060cae268b4c8ffa 1>&2 && 
        python3 main.py' in run with ID 'e2f0e8c670114c5887963cd6a1ac30f9' === 
    ```
    However, when the MLFlowLogger() is initialized for the PyTorch-Lightning 
    trainer, another `run_id` is created.
    Everything is logged to the the PyTorch-Lightning MLFlowLogger(), so the
    `run_id` created by MLflow is not necessary.
    Use this function to delete that folder.

    CAUTION:
    Starting a run in the main script to obtain the `run_id` gives the correct
    answer the first time, but any subsequent runs will have different `run_id`s.
    See https://stackoverflow.com/questions/71531665/how-to-get-run-id-when-using-mlflow-project
    """

    with mlflow.start_run():
        experiment_id = mlflow.active_run().info.experiment_id
        run_id = mlflow.active_run().info.run_id
    tracking_uri = mlflow.tracking.get_tracking_uri()
    root = tracking_uri.split('//')[-1]
    path = os.path.join(root,experiment_id, run_id)
    shutil.rmtree(path)