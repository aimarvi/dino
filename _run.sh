#!/bin/bash

#SBATCH -t 2-00:00:00
#SBATCH -c 20
#SBATCH --mem=80GB
#SBATCH --mail-user=amarvi@mit.edu
#SBATCH --mail-type=TIME_LIMIT,FAIL,END
#SBATCH --job-name=dino-in1k-test
#SBATCH --gres=gpu:2
#SBATCH --constraint=ampere
#SBATCH --output=runlog/run%j.out
#SBATCH --partition=normal

# vanilla dino training
MASTER_PORT=29501 python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --data_path /om2/group/nklab/shared/datasets/imagenet_raw/train/ --output_dir ./saved_models/vanilla/

# train on dobs 400k objects
# python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --data_path /om2/group/nklab/shared/datasets/faces_and_objects/nonface/ --output_dir ./saved_models/obj400_dino/ --meta_file ./data/dobs_obj_train.txt 
