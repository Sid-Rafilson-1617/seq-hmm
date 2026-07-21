from pathlib import Path
import numpy as np

task_file = Path("tasks/hmm_tasks.txt")
task_file.parent.mkdir(exist_ok=True)

data_dir = '/mnt/home/srafilson/code/sequences/data'
data = f"{data_dir}/maze_clipped_spike_counts_Bilat_R02_20251106.npz"

save_dir = "results_all_cells_20ms/full_fit"
Path(save_dir).mkdir(parents=True, exist_ok=True)


nStates = [251]

#restarts = range(2000)
start = 438
end = 2000
restarts = np.arange(start, end)


with task_file.open("w") as f:
    for N in nStates:
        for restart in restarts:
            seed = 100000 + 1000 * N + restart

            cmd = (
                "python -m fit_one_full "
                f"--model-params {data} "
                f"--save-dir {save_dir} "
                f"--nStates {N} "
                f"--restart {restart} "
                f"--seed {seed} "
                f"--Niters 30"

            )

            f.write(cmd + "\n")