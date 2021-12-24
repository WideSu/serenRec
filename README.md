# SEREN

## 结构

    ```
    |_dataset_
    |         |_ml-100k
    |         |_yoochoose
    |_seren_
    |       |_model_
    |       |       |_conventions.py
    |       |       |_dsr_torch.py
    |       |       |_gru4rec.py
    |       |       |_narm.py
    |       |       |_srgnn.py
    |       |_utils_
    |       |       |_data.py
    |       |       |_dataset.py
    |       |       |_functions.py
    |       |       |_metrics.py
    |       |       |_model_selections.py
    |       |_config.py
    |_test.py
    ```

## 模块介绍

test.py是目前所有程序的接口，无论跑什么程序都在这个程序基础上提交参数名就行,里面的程序内容相当于main函数

utils存放的是和程序功能相关的函数方法和类

dataset存放各类数据集，为了区分开，最好每个数据集单独建一个文件夹

model存放已经实现好的模型

注意点：

所有添加的函数注释格式为Numpy风格

model中的每个模型都会写好一个对应模型的class，需要在init方法中写好模型的参数和介绍，每个模型需要包含fit和predict两个方法，将训练过程和预测过程封装好，方便调用

data.py用来读取原始数据，目前包含两种class，Interaction用来读交互数据，Categories用来读item相关category信息的数据

dataset.py用于生成loader，因为torch的训练和预测都需要和loader相关，所以这里面放的都是和各个模型对应的loader

functions.py用来存一些方便且可能复用的函数方法

metrics.py用来计算KPI，目前可以算mrr, ndcg, hr, ILS

model_selections.py用来存放各种train-test数据集分割方法，因为会有很多种方法

config.py存放各种配置信息，以及整理test.py中通过argparse传入的各种参数

test.py中的argparse接口参数目前是直接写在里面的，如果考虑到以后框架整体完善了，没有额外参数填入时，应当将该部分一同移入config.py，或者单独命名文件args.py

conventions.py里有3个传统推荐模型，BPR的我近期会有更新


