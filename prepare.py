import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra


def code_path(path):
    code_dir = hydra.utils.get_original_cwd()
    code_dir = Path(code_dir)
    return str(code_dir / path)


def working_path(path):
    working_dir = os.getcwd()
    working_dir = Path(working_dir)
    return str(working_dir / path)



OmegaConf.register_new_resolver("code_path", code_path)
OmegaConf.register_new_resolver("working_path", working_path)