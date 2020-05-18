#!/bin/bash

data_path=../../Data/
epochs_img_head_city=10
epochs_img_all_city=30
epochs_img_head_kitti=30
epochs_img_all_kitti=0
start_epoch_kitti = 0
epochs_bev=15
start_epoch_bev=0
# start_epoch_kitti=$((epochs_img_head_city+epochs_img_all_city))


epochs_fusion=30
# start_epoch_fusion=$epochs_bev
start_epoch_fusion=15

num_summary_images_segmentation=5
num_summary_images_detection=5

train_kitti=true
train_city=false
train_bev=true
train_fusion=true
train_images_seg=true

if [ "$train_city" = true ]; then
    python Main.py --data_path $data_path \
                --train_images_seg True \
                --restore False \
                --epochs_img_head $epochs_img_head_city \
                --epochs_img_all $epochs_img_all_city \
                --batch_size 1 \
                --segmentation_kitti False \
                --segmentation_cityscapes True \
                --num_summary_images $num_summary_images_segmentation \
                --start_epoch 0 
fi

if [ "$train_kitti" = true ]; then
    python Main.py --data_path $data_path \
                --train_images_seg True \
                --restore False \
                --epochs_img_head $epochs_img_head_kitti \
                --epochs_img_all $epochs_img_all_kitti \
                --batch_size 1 \
                --segmentation_kitti True \
                --segmentation_cityscapes False \
                --num_summary_images $num_summary_images_segmentation \
                --train_fusion=False \
                --train_bev False 
fi

if [ "$train_bev" = true ]; then

    python Main.py --data_path $data_path \
                --train_bev True \
                --restore True \
                --epochs $epochs_bev \
                --start_epoch $start_epoch_bev \
                --num_summary_images $num_summary_images_detection \
                --batch_size 1 \
                --train_fusion=False \
                --train_images_seg False

    # python write_prediction_in_files.py 
  
fi

if [ "$train_fusion" = true ]; then
    python Main.py --data_path $data_path \
                --train_fusion True \
                --restore True \
                --epochs $epochs_fusion \
                --start_epoch $start_epoch_fusion \
                --num_summary_images $num_summary_images_detection \
                --batch_size 1 \
                --train_bev False \
                --train_images_seg False

fi
