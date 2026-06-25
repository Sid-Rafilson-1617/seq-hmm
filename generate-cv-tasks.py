from pathlib import Path
import numpy as np

task_file = Path("tasks/hmm_tasks.txt")
task_file.parent.mkdir(exist_ok=True)

data_dir = '/mnt/home/srafilson/code/sequences/data'
data = f"{data_dir}/maze_clipped_spike_counts_Bilat_R02_20251106.npz"

save_dir = "results/cv"
Path(save_dir).mkdir(parents=True, exist_ok=True)

folds = range(10)

nStates = [101]

restarts = range(25)
start = 24
end = 100
#restarts = np.arange(start, end)


with task_file.open("w") as f:
    for N in nStates:
        for fold in folds:
            for restart in restarts:
                seed = 100000 + 1000 * N + 100 * fold + restart

                cmd = (
                    "python -m fit_one_cv "
                    f"--data {data} "
                    f"--save-dir {save_dir} "
                    f"--fold {fold} "
                    f"--K 10 "
                    f"--r 0.8 "
                    f"--nStates {N} "
                    f"--restart {restart} "
                    f"--seed {seed} "
                    f"--Niters 20" 

                )

                f.write(cmd + "\n")