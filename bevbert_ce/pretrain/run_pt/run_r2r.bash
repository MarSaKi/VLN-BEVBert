
NODE_RANK=0
NUM_GPUS=4
outdir=../../snap_pt/r2r_ce/mlm.sap

# train
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK --master_port=$1 \
    pretrain_src/train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config run_pt/r2r_model_config_dep.json \
    --config run_pt/r2r_pretrain_habitat.json \
    --output_dir $outdir 
    