set -x

TRAIN_TEST_SPLIT=navtest

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="./data/maps"
export NAVSIM_EXP_ROOT="./logs"
export NAVSIM_DEVKIT_ROOT="."
export OPENSCENE_DATA_ROOT="./data"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
export CUDA_LAUNCH_BLOCKING=1


CHECKPOINT="./examples/models/diffusion_planner/models/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"


# 1. Set NAVSIM dataset and related environment variables
# 2. Configure torchrun (e.g., single machine: --nproc_per_node=8; adjust for multi-node)
# 3. Set agent.vlm_path and agent.checkpoint_path CHECKPOINT


uv run torchrun \
    --nproc_per_node=1 \
    $NAVSIM_DEVKIT_ROOT/examples/evaluation/run_pdm_score_recogdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    worker=sequential \
    agent=safe_copilot_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path='owl10/ReCogDrive-VLM-2B' \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    experiment_name=recogdrive_agent_eval

