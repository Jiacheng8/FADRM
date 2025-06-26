mode=fadrm
ipc=50

# Overall Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PARENT_DIR")"
source $SCRIPT_DIR/constants.sh

model_pool_dir=$Main_Data_Path/pretrained_models/${Dataset_Name}_pretrained_models/


python $PARENT_DIR/relabel.py \
    --syn-data-path $Generated_Path/generated_data/syn_data/$Dataset_Name/${mode}_ipc${ipc}\
    --fkd-path $Generated_Path/generated_data/new_labels/$Dataset_Name/${mode} \
    --teacher-model-name ResNet18\
    --model-pool-dir $model_pool_dir \
    -b 16 \
    -j 10 \
    --dataset-name $Dataset_Name \
    --eval-mode F \
    --epochs 1000 \
    --fkd-seed 42 \
    --min-scale-crops 0.08 \
    --max-scale-crops 1 \
    --use-fp16 \
    --mode 'fkd_save' \
    --mix-type 'cutmix' \