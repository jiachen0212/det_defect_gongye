# pip install -r requirements.txt
# schedctl create --name yolo7  --image "harbor.smoa.cc/public/smore_core:v2.2.0.cu10" --gpu 1 --cmd "cd /newdata/jiachen/project/competition/yolov7 && source ~/.bashrc && source activate mmdet && python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg"



schedctl create --name yolo7 --image "harbor.smoa.cc/public/smore_core:v2.2.0.cu10" --gpu 4 --cmd "cd /newdata/jiachen/project/competition/yolov7 && source ~/.bashrc && conda activate mmdet  && python -m torch.distributed.launch --nproc_per_node 4 --master_port 9533 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml"