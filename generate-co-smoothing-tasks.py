from pathlib import Path
import numpy as np

task_file = Path("tasks/co-smoothing-tasks.txt")
task_file.parent.mkdir(exist_ok=True)

data =  '/mnt/home/srafilson/code/sequences/data/maze_clipped_spike_counts_Bilat_R02_20251106.npz'
model_params = '/mnt/home/srafilson/code/sequences/co-smoothing-data/best_fit_params.npz'

save_dir = "results_all_cells_20ms/full_fit"
Path(save_dir).mkdir(parents=True, exist_ok=True)


nNeurons = 307


with task_file.open("w") as f:
    for neuron_index in range(nNeurons):


        cmd = (
            "python -m co-smooth-one "
            f"--data {data} "
            f"--model-params {model_params} "
            f"--save-dir {save_dir} "
            f"--neuron-index {neuron_index} "

        )

        f.write(cmd + "\n")