schedctl create --name yolox4 --image "harbor.smoa.cc/public/smore_core:v2.2.0.cu10" --gpu 4 --cmd "cd /newdata/jiachen/project/competition/mmdetection && source ~/.bashrc && conda activate mmdet  && python -m torch.distributed.launch --nnodes 1 --node_rank 0 --master_addr "127.0.0.1" --nproc_per_node 4 --master_port 29588 tools/train.py configs/yolox/yolox_s_8x8_300e_coco.py --seed 0 --launcher pytorch ${@:3} --work-dir output_yolox4"
