import dacite
import logging
import os
import shutil
import simple_parsing
import torch
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# Dataclass to load the data from the run
@dataclass
class TrainingCli:
    run_name: str
    no_config_edit: bool = False

def main():
    cli = simple_parsing.parse(TrainingCli)

if __name__ == "__main__":
    logging.basicConfig()
    main()