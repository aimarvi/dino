#!/bin/bash

#SBATCH -t 2-00:00:00
#SBATCH -c 20
#SBATCH --mem=80GB
#SBATCH --mail-user=amarvi@mit.edu
#SBATCH --mail-type=TIME_LIMIT,FAIL,END
#SBATCH --job-name=dino-face-r50
#SBATCH --gres=gpu:2
#SBATCH --constraint=ampere
#SBATCH --output=runlog/run%j.out
#SBATCH --partition=normal
#SBATCH --exclude=node094

# vanilla dino training
# MASTER_PORT=$RANDOM python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --data_path /om2/group/nklab/shared/datasets/imagenet_raw/train/ --output_dir ./saved_models/vanilla/

# train on dobs 400k objects
# MASTER_PORT=$RANDOM python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --data_path /om2/group/nklab/shared/datasets/faces_and_objects/nonface/ --output_dir ./saved_models/obj400_dino/ --meta_file ./data/dobs_obj_train.txt 
# MASTER_PORT=$RANDOM python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch resnet50 --data_path /om2/group/nklab/shared/datasets/faces_and_objects/nonface/ --output_dir ./saved_models/obj400_dino-r50/ --meta_file ./data/dobs_obj_train.txt 

# train on dobs 400k faces
# MASTER_PORT=$RANDOM python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --data_path /om2/group/nklab/shared/datasets/faces_and_objects/face/ --output_dir ./saved_models/face400_dino/ --meta_file ./data/dobs_face_train.txt 
# MASTER_PORT=$RANDOM python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch resnet50 --data_path /om2/group/nklab/shared/datasets/faces_and_objects/face/ --output_dir ./saved_models/face400_dino-r50/ --meta_file ./data/dobs_face_train.txt

MASTER_PORT=$RANDOM python -m torch.distributed.launch --nproc_per_node=1 main_alexnet.py --output_dir ./saved_models/alexnet_test/
