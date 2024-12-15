from colorama import Fore, Back, Style
from colorama import init as colorama_init
from omegaconf import OmegaConf
colorama_init()


def info_message(msg: str):
    print(Fore.GREEN + f"INFO: {msg}" + Style.RESET_ALL, flush=True)

def warning_message(msg: str):
    print(Fore.YELLOW + f"WARNING: {msg}" + Style.RESET_ALL, flush=True)

def error_message(msg: str):
    print(Fore.RED + f"ERROR: {msg}" + Style.RESET_ALL, flush=True)
