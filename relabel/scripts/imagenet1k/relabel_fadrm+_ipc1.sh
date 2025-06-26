mode=fadrm+
teacher=ResNet18
ipc=1

# Overall Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PARENT_DIR")"

source $SCRIPT_DIR/constants.sh
python $PARENT_DIR/relabel.py \
    --syn-data-path ${Generated_Path}/generated_data/syn_data/$Dataset_Name/${mode}_ipc${ipc}\
    --fkd-path ${Generated_Path}/generated_data/new_labels/$Dataset_Name/${mode} \
    --online \
    --multi-model \
    --model-choice ResNet18 MobileNetV2 ShuffleNetV2 AlexNet\
    -b 16 \
    -j 10 \
    --dataset-name $Dataset_Name \
    --eval-mode F \
    --epochs 300 \
    --fkd-seed 42 \
    --min-scale-crops 0.08 \
    --max-scale-crops 1 \
    --use-fp16 \
    --mode 'fkd_save' \
    --mix-type 'cutmix' \