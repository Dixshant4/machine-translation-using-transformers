""""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Author: Raeid Saqur <raeidsaqur@cs.toronto.edu>, Arvid Frydenlund <arvie@cs.toronto.edu>
Updated by: Arvie Frydenlund, Raeid Saqur and Jingcheng Niu

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
"""

import argparse
import gzip
from pathlib import Path
from typing import Union, TextIO


def smart_open(path: Path, mode: str = "r") -> Union[gzip.GzipFile, TextIO]:
    if path.suffix == ".gz":  # error?  TODO: check if this is correct
        open_ = gzip.open
        if mode[-1] != "b":
            mode += "t"
    else:
        open_ = open
    try:
        f = open_(path, mode=mode)
    except OSError as e:
        raise argparse.ArgumentTypeError(f"can't open '{path}': {e}")
    return f


def schedule_rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )
