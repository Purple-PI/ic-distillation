# Install on Jean-Zay

```sh
git clone git@github.com:Purple-PI/ic-distillation.git
cd ic-distillation
```

## Setup python env

Faire un script `set_env_a100.sh`:

```bash
module purge
module load cpuarch/amd
module load pytorch-gpu/py3/2.1.1

export PYTHONUSERBASE="/gpfsscratch/rech/vfy/uhx66kd/env/ic-distillation"
export PYTHONPATH=$PYTHONPATH:$PYTHONUSERBASE
```

Ensuite:
```sh
source set_env_a100.sh
pip install -e ./
```

## .env
Pour assurer que tout fonctionne chez chacun, on utilise des variables d'environnement pour les paths perso.

Dans un fichier `.env` dans la racine du repo définir les variables **modifier avec les paths perso**:

```
CHECKPOINT_PATH=/gpfswork/rech/vfy/uhx66kd/checkpoints/expressive-sft
DATA_PATH=CHECKPOINT_PATH=/gpfswork/rech/vfy/uhx66kd/data/expressive-sft
HF_DATASETS_CACHE=/gpfsscratch/rech/vfy/uhx66kd/datasets/.cache
RESULT_PATH=/gpfswork/rech/vfy/uhx66kd/results/expressive-sft
```


# Exemple de script Jean-Zay

```bash
#!/bin/bash
#SBATCH -p gpu_p2s
#SBATCH --job-name=training
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -A vfy@v100
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
#SBATCH --output=scripts_outputs/%x.out
source set_env.sh

set -x
export HF_DATASETS_OFFLINE="1"
export WANDB_MODE="offline"

srun python my_script.py
```

