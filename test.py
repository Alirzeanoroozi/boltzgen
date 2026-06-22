# import torch

# device_capability = torch.cuda.get_device_capability()

# print(device_capability)

import omegaconf
import hydra
from src.task.task import Task

config = omegaconf.OmegaConf.load("src/boltzgen/resources/config/design.yaml")
task = hydra.utils.instantiate(config)
if not isinstance(task, Task):
    raise TypeError("Config must be an instance of Task.")
task.run(config)