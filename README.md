# Taichi-Toys
用太极整一点有物理意义的小玩具

### 目的

最近发现太极铺张量什么的太轻松了，不用自己写CUDA kernel，还有一套能用的gui，就先用起来玩玩。

### 现有模型

#### XY model

哈密顿量为
$$
E=-J \sum_{\langle i, j\rangle} \boldsymbol{\sigma}_{i} \cdot \boldsymbol{\sigma}_{j}=-J \sum_{\langle i, j\rangle}\left(\sigma_{i x} \sigma_{j x}+\sigma_{i y} \sigma_{j y}\right)=-J \sum_{\langle i, j\rangle} \cos \left(\phi_{i}-\phi_{j}\right)
$$
计算结果