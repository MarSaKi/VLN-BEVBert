export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

flag1="--exp_name bevbert
      --run-type train
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 8
      IL.iters 10000
      IL.lr 1.5e-5
      IL.log_every 200
      IL.ml_weight 1.0
      IL.sample_ratio 0.75
      IL.decay_interval 2000
      IL.load_from_ckpt False
      IL.is_requeue True
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path [YOUR_BEST_PRETRAINED_MODEL_PATH]
      "

flag2=" --exp_name bevbert
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 4
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR ckpt/ckpt.iter9600.pth
      IL.back_algo control
      "

flag3="--exp_name bevbert
      --run-type inference
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1,2,3]
      TORCH_GPU_IDS [0,1,2,3]
      GPU_NUMBERS 4
      NUM_ENVIRONMENTS 4
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      INFERENCE.CKPT_PATH ckpt/ckpt.iter9600.pth
      INFERENCE.PREDICTIONS_FILE preds.json
      IL.back_algo control
      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      python -m torch.distributed.launch --nproc_per_node=4 --master_port $2 run.py $flag3
      ;;
esac