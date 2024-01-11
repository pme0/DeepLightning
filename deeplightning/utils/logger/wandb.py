import wandb


def init_wandb_metrics(metric_names: list, step_label: str) -> None:
    """ Define a custom x-axis metric `step_label` which is 
    synchronised with PyTprch-Lightning's `global_step`; and
    defines all other `metrics` to be plotted agains `step_label`
    """

    # define custom x-axis metric `step_label` (synchronised with PL `global_step`)
    wandb.define_metric(step_label)
        
    # initialise metrics to be plotted against `step_label`
    for m in metric_names:
        wandb.define_metric(m, step_metric=step_label)

    return step_label