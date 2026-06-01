from pathlib import Path

task_file = Path("tasks/hmm_tasks.txt")
task_file.parent.mkdir(exist_ok=True)

data = "data/simulated_data.npz"
save_dir = "results/cv"

folds = range(3)
nStates = [10, 20, 30, 40]
restarts = range(10)

with task_file.open("w") as f:
    for N in nStates:
        for fold in folds:
            for restart in restarts:
                seed = 100000 + 1000 * N + 100 * fold + restart

                cmd = (
                    "python -m fit_one "
                    f"--data {data} "
                    f"--save-dir {save_dir} "
                    f"--fold {fold} "
                    f"--K 5 "
                    f"--r 0.5 "
                    f"--nStates {N} "
                    f"--restart {restart} "
                    f"--seed {seed} "
                    f"--Niters 30"
                )

                f.write(cmd + "\n")