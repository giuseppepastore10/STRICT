#!/bin/bash


#python -m torch.distributed.launch --nproc_per_node=4 --master_port $1 train.py --config config/voc12/ZLSS.yaml --experimentid STRICT --imagedataset voc12  --spnetcheckpoint --pseudolabeling 0 -m "/home/giuseppep/SPNet/logs/voc12/SPNet_noval_correct/checkpoint_final.pth.tar" --iter 2000  --scale 2 --mirroring --batch_size 2
#python -m torch.distributed.launch --nproc_per_node=4 --master_port $1 train.py --config config/voc12/ZLSS.yaml --experimentid STRICT_it --imagedataset voc12  --spnetcheckpoint --pseudolabeling 0 -m "/home/giuseppep/STRICT/logs/voc12/STRICT/checkpoint_final.pth.tar" --iter 2000  --scale 2 --mirroring --batch_size 2
#python -m torch.distributed.launch --nproc_per_node=1 --master_port $1 eval.py --config config/voc12/ZLSS.yaml --imagedataset voc12 --model-path /home/giuseppep/STRICT/logs/voc12/STRICT/checkpoint_final.pth.tar -r gzlss
python -m torch.distributed.launch --nproc_per_node=1 --master_port $1 eval.py --config config/voc12/ZLSS.yaml --imagedataset voc12 --model-path /home/giuseppep/STRICT_it/logs/voc12/STRICT_it/checkpoint_final.pth.tar -r gzlss



python -m torch.distributed.launch --nproc_per_node=4 --master_port $1 train.py --config config/voc12/ZLSS.yaml --experimentid STRICT_bkg --imagedataset voc12  --spnetcheckpoint --pseudolabeling 0 -m "/home/giuseppep/SPNet/logs/voc12/SPNet_noval_bkg/checkpoint_final.pth.tar" --iter 2000  --scale 2 --mirroring --bkg --batch_size 2
python -m torch.distributed.launch --nproc_per_node=4 --master_port $1 train.py --config config/voc12/ZLSS.yaml --experimentid STRICT_bkg_it --imagedataset voc12  --spnetcheckpoint --pseudolabeling 0 -m "/home/giuseppep/STRICT/logs/voc12/STRICT_bkg/checkpoint_final.pth.tar" --iter 2000  --scale 2 --mirroring --bkg --batch_size 2
python -m torch.distributed.launch --nproc_per_node=1 --master_port $1 eval.py --config config/voc12/ZLSS.yaml --imagedataset voc12 --model-path /home/giuseppep/STRICT/logs/voc12/STRICT_bkg/checkpoint_final.pth.tar -r gzlss --bkg
python -m torch.distributed.launch --nproc_per_node=1 --master_port $1 eval.py --config config/voc12/ZLSS.yaml --imagedataset voc12 --model-path /home/giuseppep/STRICT/logs/voc12/STRICT_bkg_it/checkpoint_final.pth.tar -r gzlss  --bkg

python -m torch.distributed.launch --nproc_per_node=4 --master_port $1 train.py --config config/voc12/ZLSS.yaml --experimentid HP --imagedataset voc12  --spnetcheckpoint --pseudolabeling 0 -m "/home/giuseppep/SPNet/logs/voc12/SPNet_noval_correct/checkpoint_final.pth.tar" --iter 2000 --batch_size 2
python -m torch.distributed.launch --nproc_per_node=4 --master_port $1 train.py --config config/voc12/ZLSS.yaml --experimentid HP_bkg --imagedataset voc12  --spnetcheckpoint --pseudolabeling 0 -m "/home/giuseppep/SPNet/logs/voc12/SPNet_noval_bkg/checkpoint_final.pth.tar" --iter 2000  --bkg --batch_size 2
python -m torch.distributed.launch --nproc_per_node=1 --master_port $1 eval.py --config config/voc12/ZLSS.yaml --imagedataset voc12 --model-path /home/giuseppep/STRICT/logs/voc12/HP/checkpoint_final.pth.tar -r gzlss
python -m torch.distributed.launch --nproc_per_node=1 --master_port $1 eval.py --config config/voc12/ZLSS.yaml --imagedataset voc12 --model-path /home/giuseppep/STRICT/logs/voc12/HP_bkg/checkpoint_final.pth.tar -r gzlss  --bkg