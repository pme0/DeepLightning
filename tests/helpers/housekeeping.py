import shutil
import os


def clear_temporary_dir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)