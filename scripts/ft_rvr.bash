DATA_ROOT=datasets

train_alg=dagger

features=vitbase
ft_dim=768
obj_features=vitbase
obj_ft_dim=768

ngpus=4
seed=0

pref=mlm.1.mrc.1.sap.1.og.1
iter=xxx    # the selected iteration in pretraining
name=${pref}.${iter}

outdir=snap_ft/rvr/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset reverie
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --train_alg ${train_alg}
      --dagger_sample sample
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200
      --max_objects 20

      --batch_size 4
      --lr 1e-5
      --iters 40000
      --log_every 500
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0.
      --pc_order 1"

export PYTHONPATH=build:map_nav_src:$PYTHONPATH

# train -- uncomment for training
# python -m torch.distributed.run --nproc_per_node=$ngpus --master_port=$1 \
#       map_nav_src/reverie/main_nav_obj.py $flag \
#        --resume_file ${outdir}/ckpts/latest_dict --resume_optimizer \
#        --bert_ckpt_file snap_pt/rvr/${pref}/ckpts/model_step_${iter}.pt

# test
python -m torch.distributed.run --nproc_per_node=$ngpus --master_port=$1 \
      map_nav_src/reverie/main_nav_obj.py $flag  \
      --resume_file ckpts/rvr_best \
      --test --submit