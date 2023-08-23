
NODE_RANK=0
NUM_GPUS=4
task_ratio=mlm.1.mrc.1.sap.1.og.1
outdir=snap_pt/rvr/${task_ratio}

# train
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port=$1 \
    pretrain_src/train_reverie_obj.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config configs/rvr_model.json \
    --config configs/rvr_pretrain.json \
    --output_dir $outdir --task_ratio ${task_ratio}
