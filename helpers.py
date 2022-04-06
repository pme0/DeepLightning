import os
import shutil
import argparse

""" USAGE:
```
   python3 helpers.py <helper_name> --optional1 <opt1value> 
```
<helper_name> is a required positional argument; all others are
optional and should match the group parameters for the chosen
helper function.
"""

parser = argparse.ArgumentParser()
# positional argument, required
parser.add_argument("helper", type=str, choices=["clean_experiments"], help="helper function requested.")
# create optional groups
clean_experiments_group = parser.add_argument_group(title='clean_experiments() options')
# add argument to optional groups
clean_experiments_group.add_argument("--root", type=str, nargs='?', const='arg_was_not_given', help="")
# parse
args = parser.parse_args()


def clean_experiments(args: argparse.Namespace):
    """
    NOTE 
    A function `deeplightning.utils.mlflow_clean()` is now called 
    in `main.py` to clean-up the phantom folder

    PHANTOM FOLDER
    PyTorch-Lightning initializes an experiment when the logger is initialized. 
    MLflow initializes another experiment at runtime (with different run_id). 
    The corresponding phantom folder remains empty as all output is directly to 
    the experiment set-up by PyTorch-Lightning's logger.
    
    This helper function cleans-up the empty experiments.

    Arguments:
        :root: The folder ('mlruns/0/') where the run_id folders 
            exist (e.g. 'mlruns/0/26d03343d18c00e616').
    
    Example: 
    ``` 
        python3 helpers.py clean_experiments --root mlruns/0/
    ```
    will delete all empty directories at 'mlruns/0/'.
    """

    def delete_experiment(experiment: str, message: str):
        print(message, end=' ')
        shutil.rmtree(experiment)
        print("done!\n")
        return 1


    ROOT = os.path.abspath(args.root)
    
    experiments = os.listdir(ROOT)
    experiments = [os.path.join(ROOT, x) for x in experiments if os.path.isdir(os.path.join(ROOT, x))]

    status = 0

    for experiment in experiments:

        artifact_dir = os.path.join(experiment, "artifacts")
        
        if os.path.isdir(artifact_dir):

            artifacts = os.listdir(artifact_dir)

            if "config.yaml" not in artifacts or "last.ckpt" not in artifacts:
                message = "EXPERIMENT '{}': Missing config ('config.yaml') " \
                    "or checkpoint ('last.ckpt'). Deleting " \
                    "that experiment...".format(experiment)
                status += delete_experiment(experiment, message)
        else:

            message = "EXPERIMENT '{}': Missing artifact folder. " \
                "Deleting that experiment...".format(experiment)
            status += delete_experiment(experiment, message)

    if status > 0:
        print(f"Experiment file tree cleaned - deleted {status} experiments.")
    else:
        print("Experiment file tree already clean - no experiments to delete.")


if __name__ == "__main__":

    # locals() is a dictionary with current local symbol table
    locals()[args.helper](args)
