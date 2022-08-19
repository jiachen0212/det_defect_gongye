schedctl create --name 33  --image "harbor.smoa.cc/public/smore_core:v2.2.0.cu10" --gpu 1 --cmd "cd /newdata/jiachen/project/competition/mmdetection && /newdata/jiachen/miniconda3/envs/mmdet/bin/python demo.py"


# schedctl create --name 33  --image "harbor.smoa.cc/public/smore_core:v2.2.0.cu10" --gpu 1 --cmd "cd /newdata/jiachen/project/competition/mmdetection && source ~/.bashrc && source activate mmdet && pip install opencv-python-headless==4.5.3.56"

# conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch && pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.1/index.html"