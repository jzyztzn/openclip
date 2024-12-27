export MASTER_PORT=3333
export HF_ENDPOINT=https://hf-mirror.com
echo $MASTER_PORT
cd src
CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node 1 -m --master-port $MASTER_PORT open_clip_train.main \
    --train-data '../data/cc12m/cc12m-train-{0000..0575}.tar' \
    --train-num-samples 2905954 \
    --dataset-type webdataset \
    --report-to wandb \
    --wandb-project-name siglip-B-16-256 \
    --save-frequency 1 \
    --batch-size 640 \
    --precision amp \
    --workers 16 \
    --siglip \
    --model ViT-B-16-SigLIP-256 \
    --pretrained /home/models/cv/timm/ViT-B-16-SigLIP-256/open_clip_pytorch_model.bin





## test 使用12m测试集 训练ViT-L-16-SigLIP-256
# CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node 1 -m --master-port $MASTER_PORT open_clip_train.main \
#     --train-data '../data/cc12m/cc12m-train-{0000..2175}.tar' \
#     --train-num-samples 10968539 \
#     --dataset-type webdataset \
#     --report-to wandb \
#     --wandb-project-name siglip-16-256 \
#     --save-frequency 1 \
#     --batch-size 160 \
#     --precision amp \
#     --workers 16 \
#     --siglip \
#     --model ViT-L-16-SigLIP-256 \
#     --pretrained /home/models/cv/timm/ViT-L-16-SigLIP-256/open_clip_pytorch_model.bin










# export CUDA_VISIBLE_DEVICES=4  # 指定使用第 4 张 GPU
# python -m open_clip_train.main \
#     --train-data 'data/cc12m/cc12m-train-{0000..2175}.tar' \
#     --train-num-samples 10968539 \
#     --dataset-type webdataset \
#     --batch-size 1024 \
#     --precision amp \
#     --workers 16