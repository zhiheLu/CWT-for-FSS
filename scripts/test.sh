DATA=$1
SHOT=$2
GPU=$3
LAYERS=$4
SPLIT=$5

dirname="results/test/resnet-${LAYERS}/${DATA}/split_${SPLIT}"
mkdir -p -- "$dirname"

python3 -m src.test --config config_files/${DATA}.yaml \
						--opts train_split ${SPLIT} \
						     resume_weights /home/zhihelu/Research/FS_Seg/RePRI-for-Few-Shot-Segmentation/pretrained_models/${DATA}/split=${SPLIT}/model/pspnet_resnet${LAYERS}/smoothing=True/mixup=False/best.pth \
						     data_root /home/zhihelu/Research/FS_Seg/RePRI-for-Few-Shot-Segmentation/data/pascal/ \
							   batch_size_val 1 \
							   shot ${SHOT} \
							   layers ${LAYERS} \
							   cls_lr 0.1 \
							   heads 4 \
							   gpus ${GPU} \
							   test_num 1000 \
							   n_runs 5 \
						     model_dir /home/zhihelu/Research/FS_Seg/RePRI-for-Few-Shot-Segmentation/model_ckpt_back \
							   | tee ${dirname}/log_${SHOT}.txt