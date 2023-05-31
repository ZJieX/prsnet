# 推理可视化
首先公共数据集放在prsnet目录reid_inference_dataset文件中,若没有可以新建。格式为：
```
    ...
    |--prsnet
        |--reid_inference_datasets
            |--datasets_name
                |--bounding_box_test  # 测试集
                    |--xxx.jpg/xxx.png/...
                    |--......
                |--query  # 查询集
                    |--xxx.jpg/xxx.png/...
                    |--......
```

三个重要文件create_embeddings.py, get_similar.py和visual.py;
依次运行，下面详细讲述参数配置。
1. create_embeddings.py
    
    主要将公共数据集中的测试数据集提取特征并使用npy文件保存，里面存放着所有
    测试集行人图像的特征。
    
    ①. "--config_file" 参数: 使用的模型配置文件路径；

    ②. "--model_path" 参数: 训练好的模型的路径，不推荐使用绝对路径，可以使用
                          ../(表示两级目录)代替；

    ③. "--test_dir" 参数: 测试数据集路径，不推荐使用绝对路劲，可以使用
                          ../(表示两级目录)代替；

    ④. "--output" 参数: 生成的npy文件保存的路径。
    
    修改完这些参数后直接运行代码，到对应文件夹中查看结果。


2. get_similar.py

    对公共数据集中的query文件夹中的行人图像进行特征提取，且读取create_embeddings.py
    文件提取的测试集行人图像特征的npy文件，用来ReRank可视化。

    ①. "--config_file" 参数: 使用的模型配置文件路径；

    ②. "--model_path" 参数: 训练好的模型的路径，不推荐使用绝对路径，可以使用
                          ../(表示两级目录)代替；

    ③. "--query_dir" 参数: 查询数据集路径，不推荐使用绝对路劲，可以使用
                          ../(表示两级目录)代替；
    
    ④. "--gallery_data" 参数: 第1步保存npy文件的路径，即与1中的"--output"一致；

    ⑤. "--topk" 参数: 可视化前topk个结果；

    ⑥. "--output" 参数: 生成的npy文件保存的路径。

    修改好上述参数，直接运行代码即可。


3. visual.py

    将上述两个步骤ReRank成功的结果进行可视化。

    ①. im_show 参数: 是否进行可是画，默认为False；
    
    ②. file_path 参数: 经过get_similar.py生成的similer_imgs.txt文件路径；

    ③. save_path 参数: 可视化结果的保存路径，一般跟对应数据集绑定。

    修改参数后，直接运行代码即可

上述示例:
```
若此时可视化cuhk03-np数据集中的ReRank，则上述都必须同一使用cuhk03的模型以及数据集。

对于create_embeddings.py中：
    "--model_path", default="../../weights/cuhk03_best.ckpt";
    "--test_dir", default="../../centroid/inference_datasets/cuhk03-np/label/bounding_box_test/";
    "--output", default="result/cuhk03"。
对于get_similar.py中：
    "--model_path", default="../../weights/cuhk03_best.ckpt";
    "--query_dir", default="../../centroid/inference_datasets/cuhk03-np/label/bounding_box_test/";
    "--gallery_data", default="../../centroid/inference/result/cuhk03";
    "--topk", default=10;  # 只显示到Rank10
    "--output", default="result/cuhk03"。
对于visual.py中：
    file_path="result/cuhk03/similer_imgs.txt";
    save_path="result/cuhk03/cuhk03_visual/"。
    
先运行create_embeddings.py，再运行get_similar.py，最后运行visual.py
```