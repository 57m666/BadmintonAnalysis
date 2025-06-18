import numpy as np
import warnings

from ..tools.utils import clear_file

# clear the polyfit Rankwarning
warnings.simplefilter("ignore", np.RankWarning)

result_path = "res"
name_list = ["test1", "test2", "test3", "test4", "test5", "test6"]
for video_name in name_list:
    clear_file(video_name, f"{result_path}")
