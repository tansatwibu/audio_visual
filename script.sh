CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python train.py \
  --name run_av_2branch \
  --model audioVisual \
  --data_path /media/data4/home/lanlt/ducanh/dataset_av \
  --auto_split --val_ratio 0.1 \
  --batchSize 1 --nThreads 4 \
  --num_batch 100000 \
  --lr_steps 60000 90000 \
  --unet_num_layers 7 \
  --lr_unet 0.001 --lr_visual 0.0001 \
  --weighted_loss --optimizer sgd --mask_loss_type L1 \
  --coseparation_loss_weight 1 \
  --log_freq True --tensorboard True \
  --save_latest_freq 1000 \
  --validation_freq 200 --validation_batches 5 \
  --gpu_ids 0 \
  --checkpoints_dir /media/data4/home/lanlt/ducanh/final_2branch/checkpoint \
  --num_visualization_examples 4