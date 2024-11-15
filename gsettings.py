print("=" * 80)
print(
    r"""This project is authored by: Quan Wuzhou.
 ________  ___  ___  ________  ________                              
|\   __  \|\  \|\  \|\   __  \|\   ___  \                            
\ \  \|\  \ \  \\\  \ \  \|\  \ \  \\ \  \                           
 \ \  \\\  \ \  \\\  \ \   __  \ \  \\ \  \                          
  \ \  \\\  \ \  \\\  \ \  \ \  \ \  \\ \  \                         
   \ \_____  \ \_______\ \__\ \__\ \__\\ \__\                        
    \|___| \__\|_______|\|__|\|__|\|__| \|__|                        
 ___      \|__| ___  ___  ________  ___  ___  ________  ___  ___     
|\  \     |\  \|\  \|\  \|\_____  \|\  \|\  \|\   __  \|\  \|\  \    
\ \  \    \ \  \ \  \\\  \\|___/  /\ \  \\\  \ \  \|\  \ \  \\\  \   
 \ \  \  __\ \  \ \  \\\  \   /  / /\ \   __  \ \  \\\  \ \  \\\  \  
  \ \  \|\__\_\  \ \  \\\  \ /  /_/__\ \  \ \  \ \  \\\  \ \  \\\  \ 
   \ \____________\ \_______|\________\ \__\ \__\ \_______\ \_______\
    \|____________|\|_______|\|_______|\|__|\|__|\|_______|\|_______|
"""
)
print("=" * 80)

import time
from log import logger

logger.info(f"Hello!")

torch_timer = time.time()
import torch

logger.info(f"Pytorch loaded in {time.time() - torch_timer:.2f}s.")
del torch_timer

logger.info(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    logger.info(f"PyTorch CUDA Version: {torch.version.cuda}")
    logger.info(f"PyTorch CUDA Current Device: {torch.cuda.current_device()+1} ({torch.cuda.get_device_name(torch.cuda.current_device())}) / {torch.cuda.device_count()}")
    logger.info(f"PyTorch CUDNN Version: {torch.backends.cudnn.version()}")

import os, random, numpy as np, torch, time, pathlib
from typing import Union

# 随机种子
# Random seed
seed = 3407
if seed is not None:
    logger.info(f"[Settings Initiated] Global seed fixed: {seed}")
else:
    seed = random.randint(0, 100000)
    logger.warning(f"[Settings Initiated] Global seed is randomly set as: {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
del seed

# 根目录路径
# Root Path
root_path = pathlib.Path(__file__).parent.absolute()
logger.info("[Settings Initiated] Root path: %s" % os.fspath(root_path))

exps_workdir_path = root_path / "exps"
if not exps_workdir_path.exists():
    exps_workdir_path.mkdir()
logger.info("[Settings Initiated] Model save path: %s" % os.fspath(exps_workdir_path))

# 模型保存路径
# Model Save Path
model_save_path = root_path / "checkpoints"
logger.info("[Settings Initiated] Model save path: %s" % os.fspath(model_save_path))

# 输出路径
# Output Path
output_path = root_path / "outputs"
if not output_path.exists():
    output_path.mkdir()
logger.info("[Settings Initiated] Output path: %s" % os.fspath(output_path))

# mmengine 设置路径
# MMengine Deployments Path
deployment_path = root_path / "deployments"
logger.info("[Settings Initiated] MMengine deployments path: %s" % os.fspath(deployment_path))

# Dataset 文件夹路径
# Dataset Folder Path
datasets_folder_path = root_path / "data"
if not datasets_folder_path.is_dir():
    logger.critical("Dataset folder not found: %s" % os.fspath(datasets_folder_path))
    exit(-1)
logger.info("[Settings Initiated] Dataset folder path: %s" % os.fspath(datasets_folder_path))
datasets_folder_path_str = os.fspath(datasets_folder_path)

icvl_root = datasets_folder_path / "IVCL"

import mmengine
from mmengine.config import Config


def __replace_numbands__(cfg: dict, numbands):
    for k, v in cfg.items():
        if isinstance(v, dict):
            __replace_numbands__(v, numbands)
        elif v == "%numbands%":
            cfg[k] = numbands


def load_model(model_arch_name: str, input_channels: int, device: Union[str, torch.device] = "cpu"):
    model_deployment_file_path = deployment_path / "models" / f"{model_arch_name}.py"
    model_cfg = Config.fromfile(
        os.fspath(model_deployment_file_path),
        lazy_import=False,
    )
    __replace_numbands__(model_cfg, input_channels)
    model: torch.nn.Module = mmengine.MODELS.build(model_cfg.model).to(device)
    return model
