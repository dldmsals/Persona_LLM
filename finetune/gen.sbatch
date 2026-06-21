#!/bin/bash
#SBATCH -J persona_gen
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
set -e
mkdir -p logs
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona_ft
export HF_HOME="${HF_HOME:-/scratch/$USER/.cache/huggingface}"
cd "$SLURM_SUBMIT_DIR"
BASE="${BASE:-Qwen/Qwen3-4B}"
ADAPTER="${ADAPTER:-out_$(basename $BASE)}"
OUT="${OUT:-answers_ft_$(basename $BASE).jsonl}"
TESTSET="${TESTSET:-testset_100.jsonl}"
echo "[INFO] BASE=$BASE ADAPTER=$ADAPTER OUT=$OUT TESTSET=$TESTSET node=$(hostname)"
if [ "$ADAPTER" = "none" ] || [ -z "$ADAPTER" ]; then
  python -u gen_ft_answers.py --base "$BASE" --testset "$TESTSET" --out "$OUT"
else
  python -u gen_ft_answers.py --base "$BASE" --adapter "$ADAPTER" --testset "$TESTSET" --out "$OUT"
fi
