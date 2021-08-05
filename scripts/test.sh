DATA=$1
SHOT=$2
GPU=$3
LAYERS=$4
SPLIT=$5

dirname="results/test/resnet-${LAYERS}/${DATA}/split_${SPLIT}"
mkdir -p -- "$dirname"

python3 -m src.test --config config_files/${DATA}.yaml \
						--opts train_split ${SPLIT} \
							   batch_size_val 1 \
							   shot ${SHOT} \
							   layers ${LAYERS} \
							   cls_lr 0.1 \
							   heads 4 \
							   gpus ${GPU} \
							   test_num 1000 \
							   n_runs 5 \
							   | tee ${dirname}/log_${SHOT}.txt