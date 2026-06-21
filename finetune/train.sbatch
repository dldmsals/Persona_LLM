#!/bin/bash
#SBATCH -J persona_qlora
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 06:00:00

# Cluster etiquette: source ~/.bashrc, python -u, let SLURM assign the GPU (no pinning)
set -e
mkdir -p logs
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona_ft

# keep large model/HF caches off /home (use NAS scratch)
export HF_HOME="${HF_HOME:-/scratch/$USER/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
echo "[INFO] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

BASE="${BASE:-Qwen/Qwen3-0.6B}"
EPOCHS="${EPOCHS:-3}"
MAXS="${MAXS:-0}"
TAG="${TAG:-}"
OUTDIR="out_$(basename $BASE)${TAG}"
echo "[INFO] BASE=$BASE EPOCHS=$EPOCHS MAXS=$MAXS OUTDIR=$OUTDIR"

python -u train_qlora.py --base "$BASE" --epochs "$EPOCHS" --max_samples "$MAXS" \
  --data sft_data.jsonl --out "$OUTDIR"
