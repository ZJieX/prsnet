python train_base_model.py \
--config_file="reid_configs/prs.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR '/code/dataset/track/ReID/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './reid_eval_result/rps/market1501' \
SOLVER.EVAL_PERIOD 40 \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "/code/rpsnet/reid_logs/market1501/rps/train_ctl_model/best/auto_checkpoints/checkpoint_119.pth" \
REPRODUCIBLE_NUM_RUNS 1