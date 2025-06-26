EXP_NAME=fadrm+_ipc50

Start_ipc=0
End_ipc=50

# Overall Directory Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PARENT_DIR")"

source $SCRIPT_DIR/constants.sh
syn_data_dir=$Main_Data_Path/generated_data/syn_data/$Dataset_Name
patch_dir=$patch_dir/$Dataset_Name

# Create logs directory
mkdir -p $SCRIPT_DIR/logs

# Rember to change the exp name
# Script Configuration
Log_NAME="${EXP_NAME}_ipc_${Start_ipc}_${End_ipc}"
python -u $PARENT_DIR/recover.py \
    --exp-name  $EXP_NAME\
    --alpha 0.5 \
    --apply-data-augmentation \
    --teacher-model-list ResNet18 Densenet121 ShuffleNetV2 MobileNetV2\
    --pretrained-model-type online \
    --dataset-name $Dataset_Name \
    --batch-size $bs \
    --syn-data-path $syn_data_dir \
    --patch-dir $patch_dir \
    --lr 0.25 \
    --input-size-lis 200 224 200 224\
    --optimization-budgets 500 500 500 500\
    --r-bn 0.01 \
    --store-best-images \
    --ipc-start $Start_ipc \
    --ipc-end $End_ipc \
    --initialisation-method Patches \
    --patch-diff medium > $SCRIPT_DIR/logs/$Log_NAME.log 2>$SCRIPT_DIR/logs/$Log_NAME.error