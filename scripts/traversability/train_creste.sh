#!/bin/bash

# Default values
DATASET="traversability/creste_sam2elevtraverse_horizon"
MODEL="traversability/terrainnet_maxentirlcf_msfcn_sam2dynsemelev"
HORIZON="50"
TRAINER="standard"
FREEZE_WEIGHTS="True"
WEIGHTS_PATH=""
CKPT_PATH=""
MAXENT_WEIGHT="1.0"
REWARD_WEIGHT="0.0001"
ALPHA="0.0" # Vanilla maxentirl
BATCH_SIZE="10"
RESAMPLE_TRAJECTORIES="True"
ZERO_TERMINAL_STATE="False"
RUN_NAME="creste_terrainnet_dinopretrain_maxentirl_msfcn_sam2dynsemelev"

# Parse CLI arguments
while [ $# -gt 0 ]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift
      ;;
    --model)
      MODEL="$2"
      shift
      ;;
    --horizon)
      HORIZON="$2"
      shift
      ;;
    --trainer)
      TRAINER="$2"
      shift
      ;;
    --freeze_weights)
      FREEZE_WEIGHTS="$2"
      shift
      ;;
    --ckpt_path)
      CKPT_PATH="$2"
      shift
      ;;
    --weights_path)
      WEIGHTS_PATH="$2"
      shift
      ;;
    --maxent_weight)
      MAXENT_WEIGHT="$2"
      shift
      ;;
    --reward_weight)
      REWARD_WEIGHT="$2"
      shift
      ;;
    --alpha)
      ALPHA="$2"
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift
      ;;
    --split)
      SPLIT_DIR="$2"
      shift
      ;;
    --resample_trajectories)
      RESAMPLE_TRAJECTORIES="$2"
      shift
      ;;
    --zero_terminal_state)
      ZERO_TERMINAL_STATE="$2"
      shift
      ;;
    --run_name)
      RUN_NAME="$2"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done

# Adjust dataset paths based on horizon
WANDB_NAME="creste_urban_traverse_horizon${HORIZON}"
if [ -z ${SPLIT_DIR+x} ]; then
  SPLIT_DIR="data/creste/splits/3d_sam_3d_sam_dynamic_elevation_traversability_counterfactuals_hausdorff1m_horizon${HORIZON}_curvature"
fi

# Map action to different action horizons
if [ "${HORIZON}" == "50" ]; then
  ACTION_HORIZON="50"
elif [ "${HORIZON}" == "70" ]; then
  ACTION_HORIZON="50"
else
  echo "Invalid horizon: ${HORIZON}"
  exit 1
fi

# Debug print to ensure values are correct
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo "Horizon: ${HORIZON}"
echo "Trainer: ${TRAINER}"
echo "Ckpt Path: ${CKPT_PATH}"
echo "Weights Path: ${WEIGHTS_PATH}"
echo "MaxEnt Weight: ${MAXENT_WEIGHT}"
echo "Reward Weight: ${REWARD_WEIGHT}"
echo "Zero Terminal State: ${ZERO_TERMINAL_STATE}"
echo "Alpha: ${ALPHA}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Resample Trajectories: ${RESAMPLE_TRAJECTORIES}"
echo "Split Dir: ${SPLIT_DIR}"
echo "Wandb Name: ${WANDB_NAME}"

# Run the Python script with the collected parameters
python creste/train_traversability.py \
  "model=${MODEL}" \
  "dataset=${DATASET}" \
  "dataset.action_horizon=${HORIZON}" \
  "dataset.task_cfgs.3.kwargs.num_views=${HORIZON}" \
  "dataset.datasets.0.split_dir=${SPLIT_DIR}" \
  "dataset.resample_trajectories=${RESAMPLE_TRAJECTORIES}" \
  "trainer=${TRAINER}" \
  "model.run_name=${RUN_NAME}" \
  "+model.vision_backbone.freeze_weights=${FREEZE_WEIGHTS}" \
  "model.ckpt_path=\"${CKPT_PATH}\"" \
  "model.vision_backbone.weights_path=\"${WEIGHTS_PATH}\"" \
  "model.loss.0.maxent_weight=${MAXENT_WEIGHT}" \
  "model.loss.0.reward_weight=${REWARD_WEIGHT}" \
  "model.loss.0.alpha=${ALPHA}" \
  "+wandb_name=${WANDB_NAME}" \
  "model.action_horizon=${ACTION_HORIZON}" \
  "model.batch_size=${BATCH_SIZE}" \
  "model.zero_terminal_state=${ZERO_TERMINAL_STATE}"
