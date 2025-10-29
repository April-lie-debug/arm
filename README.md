# arm
本仓库对论文的伪代码部分进行验证，效果良好无错误情况，下载者可以通过更改main.py文件中的参数来验证不同的参数。具体参数含义在程序中以注释形式给出。

```utils.py```: 存放了 bradley_terry_prob 和 get_fisher_info_duel 两个与模型数学定义相关的辅助函数。

```algorithm.py```: 存放了核心的两个算法函数 estimate_beta 和 hybrid_bai，它们现在依赖于 utils.py 中的函数。

```main.py```: 存放了参数设置、模拟数据的生成（通过传入真实参数）以及运行和展示结果的逻辑。