import numpy as np

np.random.seed(42)
seeds = np.random.choice(np.arange(1, 10000), size=20, replace=False)

steps = [500]

for step in steps:
    for seed in seeds:
        with open(f"eval_bash/{step}_{seed}.sh", "w") as f:
            f.write(f"""#!/bin/bash

#SBATCH --mem=32G
#SBATCH --partition gpu
#SBATCH --gpus=h100:1
#SBATCH --job-name=rs{step}_{seed}
#SBATCH --time=0:10:00
#SBATCH --mail-type=ALL

module load miniconda
conda activate the
python blimp.py --random_seed {seed} --step {step}
        """)


with open(f"run_all_eval.sh", "w") as f:
    for step in steps:
        for seed in seeds:
            f.write(f"sbatch eval_bash/{step}_{seed}.sh"+"\n")





