from pathlib import Path
import numpy as np

task_file = Path("tasks/improved-hmm-fit-tasks.txt")
task_file.parent.mkdir(exist_ok=True)

data =  '/mnt/home/srafilson/code/sequences/data/maze_clipped_spike_counts_pyr_only_Bilat_R02_20251106.npz'
model_params = '/mnt/home/srafilson/code/sequences/co-smoothing-data/pyr_only_20ms_N251_better/best_fit_params.npz'

save_dir = '/mnt/home/srafilson/code/sequences/results_pyr_only_20ms/improved_fit/N251_better'
Path(save_dir).mkdir(parents=True, exist_ok=True)


randomInitializations = 12

    

with task_file.open("w") as f:
    for i in range(randomInitializations):

        # set the random seed for this initialization
        seed = i * 10 + i


        cmd = (
            "python -m improve_hmm_fit_one "
            f"--model-params {model_params} "
            f"--data {data} "
            f"--niters 50 "
            f"--save-dir {save_dir} "
            f"--restart {i} "
            f"--seed {seed}"
        )

        f.write(cmd + "\n")