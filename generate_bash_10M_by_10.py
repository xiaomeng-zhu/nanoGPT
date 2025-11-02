import numpy as np

np.random.seed(42)
seeds = list(np.random.choice(np.arange(1, 10000), size=20, replace=False))[:10]

# part 1: fix seed and only vary data
for subsplit in range(10):
    with open(f"training_bash_10by10/1732_{subsplit}.sh", "w") as f:
        f.write(f"""#!/bin/bash

#SBATCH --mem=32G
#SBATCH --partition gpu
#SBATCH --gpus=h100:1
#SBATCH --job-name=1732_{subsplit}
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL

module load miniconda
conda activate the
python train.py --batch_size=32 --compile=True --current_seed=1732 --subsplit={subsplit}
        """)


with open(f"run_all_part1.sh", "w") as f:
    for subsplit in range(10):
        f.write(f"sbatch training_bash_10by10/1732_{subsplit}.sh"+"\n")

# part 2: let seed vary with data
subsplits = range(10)
for seed, subsplit in zip(seeds, subsplits):
    with open(f"training_bash_10by10/{seed}_{subsplit}.sh", "w") as f:
        f.write(f"""#!/bin/bash

#SBATCH --mem=32G
#SBATCH --partition gpu
#SBATCH --gpus=h100:1
#SBATCH --job-name={seed}_{subsplit}
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL

module load miniconda
conda activate the
python train.py --batch_size=32 --compile=True --current_seed={seed} --subsplit={subsplit}
        """)


with open(f"run_all_part2.sh", "w") as f:
    for seed, subsplit in zip(seeds, subsplits):
        f.write(f"sbatch training_bash_10by10/{seed}_{subsplit}.sh"+"\n")