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
import colorful as cf

cf.use_style('solarized')

def generate_title_string():
  
    ascii_galaxy = [
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⡀⠒⠒⠦⣄⡀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠀⠀⠀⠀⢀⣤⣶⡾⠿⠿⠿⠿⣿⣿⣶⣦⣄⠙⠷⣤⡀⠀⠀⠀⠀",
        "⠀⠀⠀⣠⡾⠛⠉⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⣿⣷⣄⠘⢿⡄⠀⠀⠀",
        "⠀⢀⡾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠐⠂⠠⢄⡀⠈⢿⣿⣧⠈⢿⡄⠀⠀",
        "⢀⠏⠀⠀⠀⢀⠄⣀⣴⣾⠿⠛⠛⠛⠷⣦⡙⢦⠀⢻⣿⡆⠘⡇⠀⠀",
        "⠀⠀⠀⠀⡐⢁⣴⡿⠋⢀⠠⣠⠤⠒⠲⡜⣧⢸⠄⢸⣿⡇⠀⡇⠀⠀",
        "⠀⠀⠀⡼⠀⣾⡿⠁⣠⢃⡞⢁⢔⣆⠔⣰⠏⡼⠀⣸⣿⠃⢸⠃⠀⠀",
        "⠀⠀⢰⡇⢸⣿⡇⠀⡇⢸⡇⣇⣀⣠⠔⠫⠊⠀⣰⣿⠏⡠⠃⠀⠀⢀",
        "⠀⠀⢸⡇⠸⣿⣷⠀⢳⡈⢿⣦⣀⣀⣀⣠⣴⣾⠟⠁⠀⠀⠀⠀⢀⡎",
        "⠀⠀⠘⣷⠀⢻⣿⣧⠀⠙⠢⠌⢉⣛⠛⠋⠉⠀⠀⠀⠀⠀⠀⣠⠎⠀",
        "⠀⠀⠀⠹⣧⡀⠻⣿⣷⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⠃⠀⠀",
        "⠀⠀⠀⠀⠈⠻⣤⡈⠻⢿⣿⣷⣦⣤⣤⣤⣤⣤⣴⡾⠛⠉⠀⠀⠀⠀",
        "⠀⠀⠀⠀⠀⠀⠈⠙⠶⢤⣈⣉⠛⠛⠛⠛⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"
    ]

    title_string = r"""    * ▗▄▄▖  ▗▄▖ ▗▖   .▗▄▖ ▗▖  ▗▖▗▖  ▗▖   .▗▄▄▄▄▖ ▗▄▖  ▗▄▖ .
      ▐▌  .▐▌ ▐▌▐▌  .▐▌ ▐▌ ▝▚▞▘  ▝▚▞▘   .    ▗▞▘▐▌ ▐▌▐▌ ▐▌
    . ▐▌▝▜▌▐▛▀▜▌▐▌   ▐▛▀▜▌  ▐▌   .▐▌       ▗▞▘  ▐▌ ▐▌▐▌ ▐▌   *
      ▝▚▄▞▘▐▌ ▐▌▐▙▄▄▖▐▌ ▐▌▗▞▘▝▚▖  ▐▌  *   ▐▙▄▄▄▖▝▚▄▞▘▝▚▄▞▘ ."""

    ascii_height = len(ascii_galaxy)
    title_lines = title_string.splitlines()
    title_height = len(title_lines)

    top_padding = (ascii_height - title_height) // 2
    title_lines = [""] * top_padding + title_lines
    title_lines += [""] * (ascii_height - len(title_lines)) 

    final_output = "\n".join(f"{ghost}  {title}" for ghost, title in zip(ascii_galaxy, title_lines))
    return final_output

# Dataclass to load the data from the run
@dataclass
class TrainingCli:
    """
    Attributes:
    run_name (str): Unique name for the training run (used for logging, saving models, etc.).
    no_config_edit (bool): If True, disables manual editing of configuration after parsing.
    """
    run_name: str
    no_config_edit: bool = False

def main():
    """
    Main entry point of the script.

    Parses command-line arguments into a TrainingCli object and performs any 
    setup or execution logic required to start the training run.
    """
    # Create a minimal parser without default help sections
    parser = simple_parsing.ArgumentParser(add_help=True, description="Train your galaxy classifier.")
    
    # Add the dataclass
    parser.add_arguments(TrainingCli, dest="cli")

    # Parse arguments
    args = parser.parse_args()

    # Unpack the dataclass
    cli: TrainingCli = args.cli

    print("\n" + cf.cyan(generate_title_string()) + "\n") 
    print(f"Run name: {cli.run_name}" + "\n")
    

if __name__ == "__main__":
    # Set up basic logging configuration
    logging.basicConfig()
    main()