DATA=$1
SPLIT=$2
GPU=$3
LAYERS=$4
SHOT=$5


dirname="results/train/resnet-${LAYERS}/${DATA}/split_${SPLIT}/shot_${SHOT}"
mkdir -p -- "$dirname"
python3 -m src.train --config config_files/${DATA}.yaml \
					 --opts train_split ${SPLIT} \
						    layers ${LAYERS} \
						    gpus ${GPU} \
						    shot ${SHOT} \
						    trans_lr 0.001 \
						    heads 4 \
						    cls_lr 0.1 \
						    batch_size 1 \
						    batch_size_val 1 \
						    epochs 20 \
						    test_num 1000 \
							    | tee ${dirname}/log_${SHOT}.txt