from omegaconf import OmegaConf


def add_logger_params_to_config(cfg: OmegaConf, run_id: str, run_name: str, run_dir: str, artifact_path: str) -> OmegaConf:
    """
    """
    cfg.logger.runtime = {}
    cfg.logger.runtime.run_id = run_id
    cfg.logger.runtime.run_name = run_name
    cfg.logger.runtime.run_dir = run_dir
    cfg.logger.runtime.artifact_path = artifact_path
    return cfg