import numpy as np

np.random.seed(42)
seeds = list(np.random.choice(np.arange(1, 10000), size=20, replace=False))[:10]


subsplits = range(10)

# part 1
# for seed, subsplit in zip(seeds, subsplits):
#     with open(f"eval_bash_10by10/500_{seed}_{subsplit}.sh", "w") as f:
#         f.write(f"""#!/bin/bash

# #SBATCH --mem=32G
# #SBATCH --partition gpu
# #SBATCH --gpus=h100:1
# #SBATCH --job-name=500_{seed}_{subsplit}
# #SBATCH --time=0:10:00
# #SBATCH --mail-type=ALL

# module load miniconda
# conda activate the
# python blimp.py --random_seed {seed} --step 500 --subsplit {subsplit}
#         """)


# with open(f"run_all_eval_10by10.sh", "w") as f:
#     for seed, subsplit in zip(seeds, subsplits):
#         f.write(f"sbatch eval_bash_10by10/500_{seed}_{subsplit}.sh"+"\n")

# part 2
for subsplit in subsplits:
    with open(f"eval_bash_10by10/500_1732_{subsplit}.sh", "w") as f:
        f.write(f"""#!/bin/bash

#SBATCH --mem=32G
#SBATCH --partition gpu
#SBATCH --gpus=h100:1
#SBATCH --job-name=500_1732_{subsplit}
#SBATCH --time=0:10:00
#SBATCH --mail-type=ALL

module load miniconda
conda activate the
python blimp.py --random_seed 1732 --step 500 --subsplit {subsplit}
        """)


with open(f"run_all_eval_10by10_part2.sh", "w") as f:
    for seed, subsplit in zip(seeds, subsplits):
        f.write(f"sbatch eval_bash_10by10/500_1732_{subsplit}.sh"+"\n")





