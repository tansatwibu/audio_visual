CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python train.py \
  --name run_av_2branch \
  --model audioVisualMUSIC \
  --data_path /media/data4/home/lanlt/ducanh/dataset_av \
  --hdf5_path /media/data4/home/lanlt/ducanh/dataset_av/hdf5_split \
  --batchSize 2 --nThreads 1 \
	--continue_train	\
  --num_batch 13200 \
  --lr_steps 40000 80000 \
  --unet_num_layers 7 \
  --lr_unet 0.0001 --lr_visual 0.00001 \
  --weighted_loss --optimizer adam --mask_loss_type L1 \
  --coseparation_loss_weight 1 \
  --log_freq True --tensorboard True \
  --save_latest_freq 1000 \
  --validation_freq 200 --validation_batches 5 \
  --gpu_ids 0 \
  --checkpoints_dir /media/data4/home/lanlt/ducanh/final_2branch/checkpoint2 \
  --num_visualization_examples 4