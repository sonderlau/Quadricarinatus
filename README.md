# 小龙虾性状检测

## 关键点定位

参考 [One millisecond face alignment with an ensemble of regression trees_2014 IEEE Conference on Computer Vision and Pattern Recognition__KazemiV_SullivanJ_2014](./docs/One millisecond face alignment with an ensemble of regression trees__2014 IEEE Conference on Computer Vision and Pattern Recognition__KazemiV_SullivanJ_2014.pdf)



使用一种级联层树结构进行关键点的定位，轻量且简单。

## 长度测量

远心镜头具体参数请参考 [远心镜头规格](./远心镜头规格.pdf)，视野大小 **14.08cm*11.83cm**



目前使用的镜头为：[MER2-503-36U3MC](./docs/MER2-503-36U3MC-Datasheet-CN_V22.09.30.pdf)，分辨率为 **2448*2048**。



则单位像素的距离为：**0.057516mm*0.057763mm**。





## 项目结构



### preprocessing.py



输入标注的文件 `.csv` 或 `.tps`，以及图片文件夹。

自动分割数据集，并生成用于训练的`.xml`文件。



```bash
usage: preprocessing.py [-h] [-i] [-c] [-t]

optional arguments:
  -h, --help         show this help message and exit
  -i , --input-dir   input directory containing image files (default = images)
  -c , --csv-file    (optional) XY coordinate file in csv format
  -t , --tps-file    (optional) tps coordinate file

```



### train.py

line 12

```python
@hydra.main(version_base=None, config_path="configs", config_name="default")
```



在 `configs` 文件夹存放运行的超参数，修改`config_name`选择不同的超参数用于训练。



训练数据会上传到 [wandb](https://wandb.ai/sonderlau/Quadricarinatus) 中。



### tune_hyperparameters.py

超参数优化，使用 optuna 优化训练的超参数。



`num` ：运行轮次



