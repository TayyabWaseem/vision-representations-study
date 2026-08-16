# utils/file_utils.py

from pathlib import Path

def save_log(out_dir, filename, text):
    """ Saving logs to the results directory """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(out_dir / filename, "a") as f:
        f.write(text + "\n")