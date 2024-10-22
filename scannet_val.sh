export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export WANDB_DIR='./wandb_dir'

CURR_DBSCAN=0.95
CURR_TOPK=500
CURR_QUERY=150

# TRAIN
python main.py \
general.experiment_name="exp_scannet" \
general.project_name='exp_scannet' \
general.gpus=4 \
data.num_workers=4 \
data.batch_size=2 \
