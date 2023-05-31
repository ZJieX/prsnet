# PRSNet pre-training step
## 1. Dataset preparation
    ①. 将COCO2017和PASCAL VOC数据中的行人图像裁剪出来用于自监督学习预训练；
        进入项目目录, 修改代码文件(在此之前，将COCO数据集json格式转为xml格式):
            pretrain_dataset --> COCOaVOC --> script.py
        修改 imgs_path为对应数据集的图片路径，xmls_path为对应的xml标注信息文件，
        运行代码  python script.py 即可。
    ②. 将ReID数据集进行混合用于自监督学习预训练。
        分别使用Market1501、cuhk03-np、msmt17以及dukemtmc-reid数据集。
        进入项目目录，修改代码文件:
            pretrain_dataset --> mixReID --> script.py
        参照相应数据集格式进行路径等修改，最后运行代码:
            python script.py 即可。

## 2. Pre-training commands
    此部分是进行掩码自监督学习预训练，使用第1步准备好的数据集进行，我们不需要数据集有标签，且骨干网络无预训练模型。
    python pretrain.py --batch_size 16 --epochs 400 --model 'tiny' --mask_ratio 0.75 \
                       --data_path './pretrain_dataset/COCOaVOC' --output_dir './pretrained_model' \
                       --log_dir './pretrain_logs' --device 'cuda' --num_workers 8

    python pretrain.py --batch_size 16 --epochs 400 --model 'tiny' --mask_ratio 0.75 \
                       --data_path './pretrain_dataset/mixReID' --output_dir './pretrained_model' \
                       --log_dir './pretrain_logs' --device 'cuda' --num_workers 8
## 3. Visual testing
    预训练完后，我们使用保存好的预训练模型进行图像重建测试，并且选取重建效果较好的模型进行行人重识别下游任务;
    准备未经训练的行人测试图像放于 代码根目录/pretrain_dataset/test/ 中。
    python pretrain_test_visualization.py --model 'tiny' --device 'cuda' \
                                          --model_path './pretrained_model/checkpoint-390.pth' \
                                          --mask_ratio 0.75 --data_path './pretrain_dataset/test/'

# PRSNet + MCTL Reid training
## Train
Each Dataset and Model has its own train script.  
All train scripts are in `train_scirpts` folder with corresponding dataset name.

Example run command to train MCTL on DukeMTMC-reID
```bash
sh ./reid_train_scripts/dukemtmc/train_mctl_prs_dukemtmc.sh
```

By default all train scripts will launch 3 experiments.

## Test
To test the trained model you can use provided scripts in `train_scripts`, just two parameters need to be added:  
    
    TEST.ONLY_TEST True \  
    MODEL.PRETRAIN_PATH "path/to/pretrained/model/checkpoint.pth"
    
Example train script for testing trained CTL-Model on Market1501
```bash
python train_ctl_model.py \
--config_file="reid_configs/prs.yml" \
GPU_IDS [0] \
DATASETS.NAMES 'market1501' \
DATASETS.ROOT_DIR '/data/' \
SOLVER.IMS_PER_BATCH 16 \
TEST.IMS_PER_BATCH 128 \
SOLVER.BASE_LR 0.00035 \
OUTPUT_DIR './reid_logs/market1501/prs/' \
SOLVER.EVAL_PERIOD 40 \
TEST.ONLY_TEST True \
MODEL.PRETRAIN_PATH "./reid_logs/market1501/prs/train_ctl_model/version_0/checkpoints/epoch=119.ckpt"
```

# ReRank visual
详细请参考reid_inference/README.md 文件