# Enter the desired parameters below
ipc=  #1, 10, 50
Model_Name=  # ResNet18, ResNet50, ResNet101

# Overall Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$PARENT_DIR")"

source $SCRIPT_DIR/constants.sh

mode=fadrm

#ODP
ODP=${Generated_Data_Path}/syn_data/${Dataset_Name}/${mode}_ipc${ipc}
FKD=${Generated_Data_Path}/new_labels/${Dataset_Name}/${mode}_bs${bs}_ipc${ipc}
OPD=${Generated_Data_Path}/validate_output

mkdir -p $SCRIPT_DIR/logs
EXP_NAME="${mode}_ipc${ipc}_${Model_Name}"
python $PARENT_DIR/train_fkd.py \
    --model $Model_Name \
    --ipc $ipc \
    --exp-name $EXP_NAME \
    --original-data-path $ODP\
    --fkd-path $FKD \
    --output-dir $OPD \
    --batch-size $bs \
    --epochs 300 \
    --dataset-name $Dataset_Name \
    --gradient-accumulation-steps 2 \
    --mix-type 'cutmix' \
    --cos \
    -j 10 \
    -T 20 \
    --val-dir $val_dir > $SCRIPT_DIR/logs/$EXP_NAME.log 2>$SCRIPT_DIR/logs/$EXP_NAME.error