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


'''
class wandbLogger():
    """ Deep Lightning Logger.
    """
    def __init__(self, cfg: OmegaConf, logged_metric_names: List[str]) -> None:
        self.cfg = cfg
        self.init_wandb_server()
        self.step_label = "iteration"
        self.init_wandb_metrics(
            metric_names = logged_metric_names,
            step_label = self.step_label,
        )


    def init_wandb_server(self):
        """ Initialise W&B sever
        """
        wandb.init(
            project = self.cfg.logger.project_name,
            notes = self.cfg.logger.notes,
            tags = self.cfg.logger.tags,
        )

        # get logger runtime parameters
        self.run_id = wandb.run.id
        self.run_name = wandb.run.name
        self.run_dir = wandb.run.dir.replace("/files", "")
        self.artifact_path = wandb.run.dir

        # add logger params to config - so that it can be stored with the runtime parameters
        self.cfg = add_logger_params_to_config(
            cfg = self.cfg,
            run_id = self.run_id,
            run_name = self.run_name,
            run_dir = self.run_dir,
            artifact_path = self.artifact_path,
        )


    def init_wandb_metrics(self, metric_names: list, step_label: str) -> None:
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
    

    def log_metrics(self, metrics: dict) -> None:
        wandb.log(metrics)
'''
