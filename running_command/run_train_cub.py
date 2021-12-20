import os

string = "python train.py --name cub200_train --config_file configs/cub200.yaml --num_node 1 --tensorboard --load_path OUTPUT/pretrained_model/CC_pretrained.pth"

os.system(string)

