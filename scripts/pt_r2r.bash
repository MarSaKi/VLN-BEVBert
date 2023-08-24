
NODE_RANK=0
NUM_GPUS=4
task_ratio=mlm.5.sap.5.masksem.1
outdir=snap_pt/r2r/${task_ratio}

# train
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port=$1 \
    pretrain_src/train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config configs/r2r_model.json \
    --config configs/r2r_pretrain.json \
    --output_dir $outdir --task_ratio ${task_ratio}
