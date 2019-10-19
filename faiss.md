## Faiss



> Faiss 是 Facebook 开源的一个项目，主要为了图片相似性搜索。

#### 1.配置环境

```
docker pull docker pull illagrenan/faiss-python
```

#### 2.官方"Hello World"例子

这里仅使用 python 版，如果需要了解 C++ 版，请参考[github wiki](https://github.com/facebookresearch/faiss/wiki/Getting-started).

Faiss 总体使用过程可以分为三步：

1. 构建训练数据（以矩阵形式表达）
2. 挑选合适的 Index （Faiss 的核心部件），将训练数据 add 进 Index 中。
3. Search，也就是搜索，得到最后结果

### 构建训练数据

```
import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
```

上面的代码，生成了训练数据矩阵 xd ，以及查询数据矩阵 xq。

仅仅为了好玩，分别在 xd 和 xq 所有数据的第一个维度中添加了一个偏移量，并且偏移量随着 id 数字的增大而增大。

### 创建 Index 对象以及 add 训练数据

```
import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)
```

Faiss 是围绕 Index 对象构建的。 Faiss 也提供了许多种类的 Index， 这里简单起见，使用 ***IndexFlatL2\***： 一个蛮力L2距离搜索的索引。

所有索引都需要知道它们是何时构建的，它们运行的向量维数是多少（在我们的例子中是d）。 然后，大多数索引还需要训练阶段（training phase），以分析向量的分布。 对于IndexFlatL2，我们可以跳过此操作。

### Search

```
k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)

D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

可以对索引执行的基本搜索操作是k最近邻搜索，即：对于每个查询向量，在数据库中查找其k个最近邻居。 所以结果集应该是一个 size 为 nq * k 的矩阵。

上述代码，做了两次搜索。

第一次的查询数据，直接使用了训练数据的前5行，这样子更加有比较性。 I 和 D， 分别代表 Id 和 Distance， 也就是距离和邻居的id。结果分别如下：

```
[[  0 393 363  78]
 [  1 555 277 364]
 [  2 304 101  13]
 [  3 173  18 182]
 [  4 288 370 531]]

[[ 0.7.17517328  7.2076292   7.25116253]
 [ 0.6.32356453  6.6845808   6.79994535]
 [ 0.5.79640865  6.39173603  7.28151226]
 [ 0.7.27790546  7.52798653  7.66284657]
 [ 0.6.76380348  7.29512024  7.36881447]]
```

I 直接显示了与搜索向量最相近的 4 个邻居的 ID。 D 显示了与搜索向量之间的距离。

第二次的搜索结果如下：

```
[[ 381  207  210  477]
 [ 526  911  142   72]
 [ 838  527 1290  425]
 [ 196  184  164  359]
 [ 526  377  120  425]]

[[ 9900 10500  9309  9831]
 [11055 10895 10812 11321]
 [11353 11103 10164  9787]
 [10571 10664 10632  9638]
 [ 9628  9554 10036  9582]]
```

------

# 进一步

## 1. 如何更快

### 理论基础

为了加快搜索速度，我们可以按照一定规则或者顺序将数据集分段。 我们可以在 d 维空间中定义 Voronoi 单元，并且每个数据库向量都会落在其中一个单元。 在搜索时，查询向量 x， 可以经过计算，算出它会落在哪个单元格中。 然后我们只需要在这个单元格以及与它相邻的一些单元格中，进行与查询向量 x 的比较工作就可以了。

（这里可以类比 HashMap 的实现原理。训练就是生成 hashmap 的过程，查询就是 getByKey 的过程）。

上述工作已经通过 ***IndexIVFFlat\*** 实现了。 这种类型的索引需要一个训练阶段，可以对与数据库矢量具有相同分布的任何矢量集合执行。 在这种情况下，我们只使用数据库向量本身。

IndexIVFFlat 同时需要另外一个 Index： quantizer， 来给 Voronoi 单元格分配向量。 每个单元格由质心（centroid）定义。 找某个向量落在哪个 Voronoi 单元格的任务，就是一个在质心集合中找这个向量最近邻居的任务。 这是另一个索引的任务，通常是 ***IndexFlatL2\***。

这里搜索方法有两个参数：nlist（单元格数量），nprobe （一次搜索可以访问的单元格数量，默认为1）。 搜索时间大致随着 nprobe 的值加上一些由于量化产生的常数，进行线性增长。

### 代码

```
nlist = 100
k = 4
quantizer = faiss.IndexFlatL2(d)  # 另外一个 Index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
       # 这里我们指定了 METRIC_L2, 默认它执行 inner-product 搜索。
assert not index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)                  # add may be a bit slower as well
D, I = index.search(xq, k)     # actual search
print(I[-5:])                  # neighbors of the 5 last queries
index.nprobe = 10              # default nprobe is 1, try a few more
D, I = index.search(xq, k)
print(I[-5:])                  # neighbors of the 5 last queries
```

### 运行结果

当 nprobe=1 的输出：

```
[[ 9900 10500  9831 10808]
 [11055 10812 11321 10260]
 [11353 10164 10719 11013]
 [10571 10203 10793 10952]
 [ 9582 10304  9622  9229]]
```

结果与蛮力搜索类似，但不完全相同（见上文）。 这是因为一些结果不在完全相同的 Voronoi 单元格中。 因此，访问更多的单元格可能会有用。

把 nprobe 上升到10的结果：

```
[[ 9900 10500  9309  9831]
 [11055 10895 10812 11321]
 [11353 11103 10164  9787]
 [10571 10664 10632  9638]
 [ 9628  9554 10036  9582]]
```

这个就是正确的结果。 请注意，在这种情况下获得完美结果仅仅是数据分布的工件，因为它在x轴上具有强大的组件，这使得它更容易处理。 nprobe 参数始终是调整结果的速度和准确度之间权衡的一种方法。 设置nprobe = nlist会产生与暴力搜索相同的结果（但速度较慢）。

## 2. 减少内存占用

### 理论基础

IndexFlatL2 和 IndexIVFFlat 都会存储所有的向量。 为了扩展到非常大的数据集，Faiss提供了一些变体，它们根据产品量化器（product quantizer）压缩存储的矢量并进行有损压缩。

矢量仍然存储在Voronoi单元中，但是它们的大小减小到可配置的字节数m（d必须是m的倍数）。

压缩基于[Product Quantizer](https://hal.inria.fr/file/index/docid/514462/filename/paper_hal.pdf)， 其可以被视为额外的量化水平，其应用于要编码的矢量的子矢量。

在这种情况下，由于矢量未精确存储，因此搜索方法返回的距离也是近似值。

### 代码

```
nlist = 100
m = 8                             # number of bytes per vector
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                    # 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
index.add(xb)
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
index.nprobe = 10              # make comparable with experiment above
D, I = index.search(xq, k)     # search
print(I[-5:])
```

### 结果

结果看起来时这个样子的：

```
[[   0  608  220  228]
 [   1 1063  277  617]
 [   2   46  114  304]
 [   3  791  527  316]
 [   4  159  288  393]]

[[ 1.40704751  6.19361687  6.34912491  6.35771513]
 [ 1.49901485  5.66632462  5.94188499  6.29570007]
 [ 1.63260388  6.04126883  6.18447495  6.26815748]
 [ 1.5356375   6.33165455  6.64519501  6.86594009]
 [ 1.46203303  6.5022912   6.62621975  6.63154221]]
```

我们可以观察到我们正确找到了最近邻居（它是矢量ID本身），但是矢量与其自身的估计距离不是0，尽管它明显低于到其他邻居的距离。 这是由于有损压缩造成的。

这里我们将64个32位浮点数压缩为8个字节，因此压缩因子为32。

搜索真实查询时，结果如下所示：

```
[[ 9432  9649  9900 10287]
 [10229 10403  9829  9740]
 [10847 10824  9787 10089]
 [11268 10935 10260 10571]
 [ 9582 10304  9616  9850]]
```

它们可以与上面的IVFFlat结果进行比较。 对于这种情况，大多数结果都是错误的，但它们位于空间的正确区域，如10000左右的ID所示。 实际数据的情况更好，因为：

- 统一数据很难索引，因为没有可用于聚类或降低维度的规律性
- 对于自然数据，语义最近邻居通常比不相关的结果更接近。

### 简化索引构建

由于构建索引可能变得复杂，因此有一个工厂函数在给定字符串的情况下构造它们。 上述索引可以通过以下简写获得：

```
index = faiss.index_factory(d, "IVF100,PQ8")
```

把”PQ8”替换为“Flat”，就可以得到一个 IndexFlat。

当需要预处理（PCA）应用于输入向量时，工厂就特别有用。 例如，要使用PCA投影将矢量减少到32D的预处理时，工厂字符串应该时：”PCA32,IVF100,Flat”。

## 3. 使用 GPU

### 获取单个 GPU 资源

```
res = faiss.StandardGpuResources()  # use a single GPU
```

### 使用GPU资源构建GPU索引

```
# build a flat (CPU) index
index_flat = faiss.IndexFlatL2(d)
# make it into a gpu index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
```

多个索引可以使用单个GPU资源对象，只要它们不发出并发查询即可。

### 使用

获得的GPU索引可以与CPU索引完全相同的方式使用：

```
gpu_index_flat.add(xb)         # add vectors to the index
print(gpu_index_flat.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = gpu_index_flat.search(xq, k)  # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

### 使用多个 GPU

使用多个GPU主要是声明几个GPU资源。 在python中，这可以使用index_cpu_to_all_gpus帮助程序隐式完成。

```
ngpus = faiss.get_num_gpus()

print("number of GPUs:", ngpus)

cpu_index = faiss.IndexFlatL2(d)

gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    cpu_index
)

gpu_index.add(xb)              # add vectors to the index
print(gpu_index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = gpu_index.search(xq, k) # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

# 在项目中的实现

以上都来自github上faiss 项目的wiki，可能读下来，对如何在项目中使用还是不明所以。

所以我之后又特意做了个总结，记录一下Faiss实际项目的操作经验，详情参考[这里](https://waltyou.github.io/Faiss-In-Project/)。

------

# 参考链接

1. [faiss wiki](https://github.com/facebookresearch/faiss/wiki)

在上一篇，我们知道了 Index 是 Faiss 最重要的一部分，来详细了解它们。

------

------

# Faiss Indexs 的进一步了解

# 挑一个合适的 Index

Faiss 提供了很多 Index，那么如何根据实际情况选择 Index 呢？

可以以下根据几个必要的问题，来找到自己合适的 Index 类型。

需要注意：

- 以下都通过 index_factory 字符串来表示不同 Index
- 如果需要参数，使用相应的 ParameterSpace 参数

## 1. 是否需要精确的结果？

### 那就使用 “Flat”

可以保证精确结果的唯一索引是 IndexFlatL2。

它为其他索引的结果提供基线。

它不压缩向量，但不会在它们之上增加开销。 它不支持添加id（add_with_ids），只支持顺序添加，因此如果需要 add_with_ids，请使用“IDMap,Flat”。

***是否支持 GPU***： yes

## 2. 是否关心内存？

请记住，所有Faiss索引都存储在RAM中。 以下的这些考虑是，如果我们不需要精确的结果而 RAM 又是限制因素，那么，我们就需要在内存的限制下，来优化精确-速度比（precision-speed tradeoff）。

### 如果不需要关心内存：“HNSWx”

如果你有大量的RAM或数据集很小，HNSW 是最好的选择，它是一个非常快速和准确的索引。 x 的范围是[4, 64]，它表示了每个向量的链接数量，越大越精确，但是会使用越多的内存。

速度-精确比（speed-accuracy tradeoff）可以通过 efSearch 参数来设置。 每个向量的内存使用是情况是（d * 4 + x * 2 * 4 ）。

HNSW 只支持顺序添加（不是add_with_ids），所以在这里再次使用 IDMap 作为前缀（如果需要）。 HNSW 不需要训练，也不支持从索引中删除矢量。

***是否支持 GPU***： no

### 如果有些担心内存：“…,Flat”

“…” 表示必须事先执行数据集的聚类（如下所示）。 在聚类之后，“Flat”只是将向量组织到不同桶中，因此它不会压缩它们，存储大小与原始数据集的大小相同。 速度和精度之间的权衡是通过 nprobe 参数设置的。

***是否支持 GPU***： yes（但是聚类方法也需要支持GPU）

### 如果相当关心内存：“PCARx,…,SQ8”

如果存储整个向量太昂贵，则执行两个操作：

- 使用尺寸为x的PCA以减小尺寸
- 每个矢量分量的标量量化为1个字节。

因此，总存储量是每个向量 x 个字节。

***是否支持 GPU***： no

### 如果非常关心内存：“OPQx_y,…,PQx”

PQx 代表了通过一个product quantizer压缩向量为 x 字节。 x 一般 <= 64，对于较大的值，SQ 通常是准确和快速的。

OPQ 是向量的线性变换，使其更容易压缩。 y是一个维度：

- y是x的倍数（必需）
- y <= d，d为输入向量的维度（最好）
- y <= 4*x（最好）

***是否支持 GPU***： yes（注意：OPQ转换是在软件中完成的，但它不是性能关键）

## 3. 数据集有多大

这个问题用于选择聚类选项（就是上面的那些”…“）。 数据集聚集到存储桶中，在搜索时，只访问了一小部分存储桶（nprobe 个存储桶）。 聚类是在数据集矢量的代表性样本上执行的，通常是数据集的样本。 我们指出该样本的最佳大小。

### 如果向量数量低于1百万：”…,IVFx,…”

当数据集的数量为 N 时，那么 x 应该处于 4 * sqrt(N) 和 16 * sqrt(N) 之间。 这只是用k-means聚类向量。 你需要 30 * x 到 256 * x 的矢量进行训练（越多越好）。

***是否支持 GPU***： yes

### 如果向量数量位于1百万-1千万之间：”…,IMI2x10,…”

（这里x是文字x，而不是数字）

IMI在训练向量上执行具有2^10个质心的 k-means，但它在向量的前半部分和后半部分独立地执行。 这将簇的数量增加到 2^(2 * 10)。您将需要大约64 * 2 ^ 10个向量进行训练。

***是否支持 GPU***： no

### 如果向量数量位于1千万-1亿之间：”…,IMI2x12,…”

与上面相同，将10替换为12。

***是否支持 GPU***： no

### 如果向量数量位于1亿-10亿之间：”…,IMI2x14,…”

与上面相同，将10替换为14。

***是否支持 GPU***： no

------

# 基本的 Index

## 1. 方法摘要

详情参考 [这里](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

## 2. 单元-探测（Cell-probe） 方法

以失去保证以找到最近邻居为代价来加速该过程的典型方法是采用诸如k均值的分区技术。 相应的算法有时被称为 cell-probe 方法：

我们使用基于多探测的基于分区的方法（可以联想到best-bin KD-tree的一种变体）。

- 特征空间被划分为 ncells 个单元格。
- 由于散列函数（在k均值的情况下，对最靠近查询的质心的分配），数据库向量被分配给这些单元中的一个，并且存储在由ncells反向列表形成的反向文件结构中。
- 在查询时，会选择一组 nprobe 个的反向列表
- 将查询与分配给这些列表的每个数据库向量进行比较

这样做，只有一小部分数据库与查询进行比较：作为第一个近似值，这个比例是 nprobe / ncells，但请注意，这个近似值通常被低估，因为反向列表的长度不相等。 当未选择给定查询的最近邻居的单元格时，将显示失败案例。

在C++中，相应的索引是索引IndexIVFFlat。

构造函数将索引作为参数，用于对反转列表进行赋值。 在该索引中搜索查询，并且返回的向量id（s）是应该被访问的反向列表。

## 3. 具有平坦索引作为粗量化器的单元探测方法

一般的，我们使用一个 Flat index 作为粗糙量化。 IndexIVF 的训练方法给 flat index 添加了质心。 nprobe 的值在搜索时设置（对调节速度-准确比很管用）。

**注意**： 根据经验，n 表示要被索引的点的数量， 一般确定合适质心数量的方法是在“分配向量到质心的开销（如果是纯粹的kmeans：ncentroids * d）” 和 “解析反转列表时执行的精确距离计算的数量（按照 kprobe / ncells * n * C 的顺序，其中常量 C 考虑了列表的不均匀分布， 以及当使用质心批处理时单个矢量比较更有效的事实，比如 C = 10）”之间找平衡。

这导致了许多质心的数量都遵循 ncentroids = C * sqrt（n）。

注意：在引擎盖下，IndexIVFKmeans 和 IndexIVFSphericalKmeans 不是对象，而是返回正确设置的 IndexIVFFlat 对象的函数。

警告：分区方法容易受到维数的诅咒。对于真正高维数据，实现良好的召回需要具有非常多的probe。

### 与LSH的关系

最流行的单元探测方法可能是最初的Locality Sensitive Hashing方法，称为[E2LSH] (http://www.mit.edu/~andoni/LSH/).

然而，这种方法及其衍生物有两个缺点：

- 它们需要大量的散列函数（=分区）来实现可接受的结果，从而导致大量额外的内存。内存并不便宜。
- 散列函数不适合输入数据。这对于证明很有用，但在实践中会导致选择结果不理想。

## 4. 二进制代码

在Python中，构造（改进的）LSH索引及搜索如下：

```
n_bits = 2 * d
lsh = faiss.IndexLSH (d, n_bits)
lsh.train (x_train)
lsh.add (x_base)
D, I = lsh.search (x_query, k)
```

d 代表输入向量的维度，nbits 代表存储每个向量的位数，

**注意**：算法虽然不是vanilla-LSH，但却是更好的选择。 如果 n_bits <= d，则使用正交投影仪组，如果n_bits > d，则使用紧密帧。

## 5. 基于量化的方法

基于产品量化的索引由关键字PQ标识。 例如，基于产品量化的最常见索引声明如下：

```
m = 16                                   # number of subquantizers
n_bits = 8                               # bits allocated per subquantizer
pq = faiss.IndexPQ (d, m, n_bits)        # Create the index
pq.train (x_train)                       # Training
pq.add (x_base)                          # Populate the index
D, I = pq.search (x_query, k)            # Perform a search
```

位数n_bits必须等于8,12或16. 维度d应为m的倍数.

## 6. 具有PQ细化的反转文件

IndexIVFPQ 可能是大规模搜索最有用的索引结构。

```
coarse_quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ (coarse_quantizer, d,
                          ncentroids, code_size, 8)
index.nprobe = 5
```

有关ncentroids的设置，请参阅有关IndexIVFFlat的章节。 code_size 通常是4到64之间的2的幂。 对于 IndexPQ，d 应该是 m 的倍数。

------

# 预处理和后处理

预处理和后处理用于：重新映射矢量ID，将变换应用于数据，并使用更好的索引重新对搜索结果进行排名。

## 1. Faiss ID映射

默认情况下，Faiss 为添加到索引的向量分配顺序 id。 本页介绍如何将其更改为任意ID。

一些Index类实现了 add_with_ids 方法，除了向量之外，还可以提供64位向量id。 在搜索时，类将返回存储的id而不是初始向量。

### IndexIDMap

此索引封装了另一个索引，并在添加和搜索时转换ID。它维护一个带有映射的表。

举个例子：

```
index = faiss.IndexFlatL2(xb.shape[1]) 
ids = np.arange(xb.shape[0])
index.add_with_ids(xb, ids)  # this will crash, because IndexFlatL2 does not support add_with_ids
index2 = faiss.IndexIDMap(index)
index2.add_with_ids(xb, ids) # works, the vectors are stored in the underlying index
```

### IndexIVF 中的 IDs

IndexIVF 子类始终存储矢量ID。 因此，IndexIDMap 的附加表是浪费空间。 IndexIVF 本身提供 add_with_ids。

## 2. 预处理数据

在索引之前转换数据通常很有用。 转换类继承 VectorTransform。 VectorTransform 将变换应用于输入向量（维度为d_in）并输出大小为d_out的向量。

| 转换类型     | 类名                     | 注释                                                         |
| :----------- | :----------------------- | :----------------------------------------------------------- |
| 随机回转     | RandomRotationMatrix     | 在IndexPQ 或 IndexLSH 中建立索引之前重新平衡向量的组件很有用 |
| 重新映射维度 | RemapDimensionsTransform | 减少或增加向量的大小，因为索引具有首选维度，或者在维度上应用随机排列 |
| PCA          | PCAMatrix                | 减少维数                                                     |
| OPQ rotation | OPQMatrix                | OPQ对输入向量应用旋转，使其更适合PQ编码。有关详细信息，请参阅[这里](https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html) |

如果有意义的话，可以使用方法 train 从一组矢量训练变换。它们可以 apply 于一组矢量。

索引可以包装在 IndexPreTransform 索引中，以便映射透明地进行，并且 train 与 index train 集成在一起。

## 3. 示例：应用PCA以减少尺寸数

### 使用 IndexPreTransform

例如，如果输入向量是2048D，并且必须减少到16个字节，那么使用PCA减少它们是有意义的。

```
# the IndexIVFPQ will be in 256D not 2048
coarse_quantizer = faiss.IndexFlatL2(256)
sub_index = faiss.IndexIVFPQ (coarse_quantizer, 256, ncoarse, 16, 8)
# PCA 2048->256
# also does a random rotation after the reduction (the 4th argument)
pca_matrix = faiss.PCAMatrix (2048, 256, 0, True) 

#- the wrapping index
index = faiss.IndexPreTransform (pca_matrix, sub_index)

# will also train the PCA
index.train(...)
# PCA will be applied prior to addition
index.add(...)
```

## 4. 示例：增加尺寸数

有时通过在向量中插入零来增加维数d是有用的。 这对以下内容非常有用：

- 使d为4的倍数，这就是距离计算的优化
- 使d为M的倍数，其中M是PQ的大小

```
# input is in dimension d, but we want a multiple of M
d2 = int((d + M - 1) / M) * M
remapper = faiss.RemapDimensionsTransform (d, d2, true)
# the index in d2 dimensions  
index_pq = faiss.IndexPQ(d2, M, 8)  

# the index that will be used for add and search 
index = faiss.IndexPreTransform (remapper, index_pq)
```

## 5. IndexRefineFlat：重新排名搜索结果

在查询向量时，使用实际距离计算对搜索结果进行重新排序可能是有用的。 以下示例使用 IndexPQ 搜索索引，然后通过计算实际距离重新排列第一个结果：

```
q = faiss.IndexPQ(d, M, nbits_per_index)
rq = faiss.IndexRefineFlat(q)
rq.train (xt)
rq.add (xb)
rq.k_factor = 4
D, I = rq.search(xq, 10)
```

搜索功能将从 IndexPQ 中获取4 * 10个最近邻居，然后计算每个结果的实际距离，并保留10个最佳结果。 请注意，IndexRefineFlat 必须存储完整的向量，因此它不具有内存效率。

## 6. IndexShards：组合来自多个索引的结果

当数据集分布在多个索引上时，可以通过它们调度查询，并将结果与 IndexShards 结合使用。 如果索引分布在多个GPU上并且查询可以并行完成，这也很有用. 请参阅在 GpuClonerOptions中 将 shards 设置为 true 的 index_cpu_to_gpus 。

------

# 索引IO，索引工厂，克隆和超参数调整

Faiss索引通常是复合索引，对于各个索引类型来说，这并不容易操作。 此外，它们具有许多参数，并且通常很难找到给定用例的最佳结构。 因此，Faiss提供了一个高级接口来批量操作索引并自动探索参数空间。

## 1. I/O 和深度复制索引

以下所有功能都会生成深层副本，即您无需关心对象所有权。

I/O 函数有：

- write_index(index, “large.index”): 将给定索引写入文件large.index
- Index * index = read_index(“large.index”): 读取一个 index 文件

克隆函数有：

- Index* index2 = clone_index(index)： 返回一个深层克隆
- Index * index_cpu_to_gpu = index_cpu_to_gpu(resource, dev_no, index)：深度复制索引，并将可以GPU-ified的部分放在GPU上。 例如，对于包含 IndexIVFP 的 IndexPreCompute，只有 IndexIVFPQ 将被复制到 GPU。可以提供第4个参数来设置复制选项（float16精度等）
- Index *index_gpu_to_cpu = index_gpu_to_cpu(index): 向另一个方向复制。
- index_cpu_to_gpu_multiple: 使用 IndexShards 或 IndexProxy 将索引复制到多个 GPU。请注意，IndexShards 目前不支持添加索引。

## 2. 索引工厂

index_factory 解释字符串以生成复合 Faiss 索引。 该字符串是一个逗号分隔的列表，最多包含3个组件：预处理组件，倒置文件和细化组件。

对于矢量预处理，index_factory支持：

- PCA：“PCA64”意味着通过PCA将尺寸减小到64D（用PCAMatrix实现）。 “PCAR64” 表示执行PCA，随后是随机旋转。跟随Flat索引时很有用。
- OPQ：“OPQ16”对应于准备 PQ 或 IVFPQ 至16字节代码的 OPQMatrix。使用OPQMatrix实现。当跟随 PQ 但训练缓慢时，对于某些类型的数据很有用。

对于倒置文件，它支持：

- 平坦指数“IVF4096”意味着使用平坦的粗量化器IndexFlatL2构建大小为4096的倒置文件。
- 反向多索引“IMI2x8”意味着使用2x8位的反向多索引量化器构建大小为 2^(2*8) 的反向文件（使用MultiIndexQuantizer）。
- 如果您不打算使用反向文件但需要add_with_ids，则可以设置“IDMap”以生成封装索引的IndexIDMap。

对于细化，它支持：

- 用“Flat”存储完整的向量。使用IndexFlat或IndexIVFFlat实现
- 使用“PQ16”，使用16字节的PQ细化代码。使用IndexPQ或IndexIVFPQ实现
- 使用“PQ8 + 16”时，仅在反向文件选项之后工作，使用存储在反向索引中的8个字节和16个字节的细化。使用IndexIVFPQR实现

### 例子

index = index_factory(128, “PCA80,Flat”): 产生128D向量的索引，通过PCA将它们转换为80D，然后进行穷举搜索。

index = index_factory(128, “OPQ16_64,IMI2x8,PQ8+16”): 获取128D向量，将OPQ变换应用于64D中的16个块，使用2x8位的反向多索引（= 65536个反向列表），并使用大小为8的PQ进行精炼，然后是16个字节。

### 指南

除了数据向量在输入上已经具有块方结构（例如，SIFT）之外，当跟随PQ时， OPQ是有用的。

IVFFlat 通常是最快的选择，因此如果内存有限，PQ变体很有用。

IVF的相关大小介于sqrt（n）和16 * sqrt（n）之间。

对于1M矢量，IMI的相关比特数大约为8-10，对于1G向量，大小为12-14。请注意，IMI存在内存开销，与2 ^（2 * nbits）成比例。

## 3. 自动调整运行时参数

在本节中，我们将重点关注Faiss索引的一个子集，因为它们对于在各种约束下索引1M到1G向量最有用。

索引的参数可以分为：

- 构建索引时必须设置的构建时参数
- 在执行搜索之前可以调整的运行时参数

对运行时参数执行自动调整。调整的参数是：

| key          | 索引名称            | 运行时参数 | 注释                                      |
| :----------- | :------------------ | :--------- | :---------------------------------------- |
| IVF*, IMI2x* | IndexIVF*           | nprobe     | 调整速度 - 精度权衡的主要参数             |
| IMI2x*       | IndexIVF            | max_codes  | 对于IMI很有用，它通常具有不平衡的反向列表 |
| PQ*          | IndexIVFPQ, IndexPQ | ht         | 多义的 Hamming threshold                  |
| PQ*+*        | IndexIVFPQR         | k_factor   | 确定验证了多少结果向量                    |

自动调整探索速度精度空间并保持最佳点。

### AutoTuneCriterion 对象

AutoTuneCriterion包含搜索的地面实况结果并评估搜索结果。 它返回0到1之间的性能度量。 目前有1-recall@R 和 R-recall@R（又名：交叉）标准的实现。

### OperatingPoints 对象

该对象将所有操作点存储为（性能，搜索时间，parameter_set_id）元组，并选择最佳的操作点。 如果没有其他操作点可以更快地获得至少相同的性能，则操作点是最佳的。

### ParameterSpace 对象

ParameterSpace对象扫描给定索引的可调参数，并为每个参数构建一个可能值的表。

参数空间随参数的数量呈指数级增大。我们以随机顺序探索它。

我们应用启发式修剪探索空间。对于每个参数，较高的值更慢且更准确。 因此，给予两个元组 a 和 b 以及他们的局部顺序，如果 a >= b， 那么 a 就比 b 慢和更准确。 因此，在评估一组参数之前，我们检查是否存在（b，c）的组合（ a> = b和c> = a）。 如果（perf_c，time_b）不是最佳操作点，那么我们跳过实验，因为它保证产生非最佳操作点。 这很快就会修复一系列实验来实现。

一组参数表示为人类可读的字符串（“nprobe = 2，ht = 52”）和索引（cno）。

### 优化的可靠性

有3个参数影响自动调整（或手动调整，顺便说一句）的可靠性：

- 查询集的敏感性和标准。
- 数据集应该有足够的查询点，至少1000，最好是10000，对参数设置足够敏感。
- 时间的可靠性。时间以挂钟时间来衡量。这在未占用的GPU或1个CPU线程上最可靠。 它在多线程中不太可靠。数据集越小，时序越不可靠。低于1s的多线程测量不可靠。

------

# 对索引的特殊操作

这里有一些操作，并不是适用于所有 index。

## 1. 从索引重建向量

方法 reconstruct 和 reconstruct_n 从给定其ID的索引重建一个或多个向量。

例子参考：[test_index_composite.py](https://github.com/facebookresearch/faiss/blob/master/tests/test_index_composite.py#L35).

支持：IndexFlat， IndexIVFFlat（要先调用 make_direct_map），IndexIVFPQ（一样），IndexPreTransform（如果底层变换支持它）。

## 2. 从索引中删除元素

remove_ids 方法从索引中删除向量子集。 它需要为索引中的每个元素调用一个 IDSelector 对象来决定是否应该删除它。 IDSelectorBatch将为索引列表执行此操作。 Python接口有效地构建了这个。

注意，由于它对整个数据库进行了传递，因此只有在需要删除大量向量时才有效。

例子：[test_index_composite.py](https://github.com/facebookresearch/faiss/blob/master/tests/test_index_composite.py#L25)

支持：IndexFlat, IndexIVFFlat, IndexIVFPQ, IDMap

## 3. 范围搜索

方法range_search返回查询点周围半径内的所有向量（而不是k最近的向量）。 由于每个查询的结果列表大小不同，因此必须特别处理：

- 在C++ 中，它以预先分配的 [RangeSearchResult](https://github.com/facebookresearch/faiss/blob/master/AuxIndexStructures.h#L35) 结构返回结果
- 在 Python 中，结果以一个一维数组 lims, D, I 的元组形式返回。 搜索 i 的结果在 I[lims[i]:lims[i+1]], D[lims[i]:lims[i+1]] 。

支持：IndexFlat, IndexIVFFlat (CPU only).

## 4. 拆分和合并索引

- merge_from 将另一个索引复制到此并在运行时解除分配。 您可以将 ivflib::merge_into 用于 包含在预转换中的IndexIVF。
- copy_subset_to 将此代码的子集复制到另一个索引。 用法示例：[在GPU上构建索引，然后将其移至CPU](https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_1bn.py#L541)

这些函数仅针对 IndexIVF 子类实现，因为它们主要用于大型索引。

# Faiss 中的线程与异步调用

来了解一下 Faiss 中的线程与异步调用。

------

------

# 线程安全

Faiss CPU索引对于并发搜索以及不更改索引的其他操作是线程安全的。 多线程使用改变索引的函数需要实现互斥。

即使对于只读函数，Faiss GPU索引也不是线程安全的。 这是因为GPU Faiss的 StandardGpuResources 不是线程安全的。 必须为主动运行GPU Faiss索引的每个线程创建一个StandardGpuResources对象。 多个GPU索引由单个CPU线程管理并共享相同的StandardGpuResources（实际这是应该的，因为它们可以使用相同的GPU内存临时区域）。 单个GpuResources对象可以支持多个设备，但只能支持单个调用CPU线程。 多GPU Faiss（通过index_cpu_to_gpu_multiple获得）在内部运行来自不同线程的不同GPU索引。

------

# 内部线程

Faiss 本身有几种不同的内部线程。 对于 CPU Faiss，索引（训练，添加，搜索）的三个基本操作是内部多线程的。 线程是通过 OpenMP 和多线程 BLAS 实现完成的。 Faiss 没有设置线程数。 调用者可以通过环境变量 OMP_NUM_THREADS 调整此数字，也可以随时调用 omp_set_num_threads（10）。 这个函数在 Python 中通过 faiss 提供。

对于 add 和 search 函数，线程在矢量上。 这意味着查询或添加单个向量不是多线程的。

------

# search 的性能

当批量提交查询时，获得 QPS 方面的最佳性能。

如果逐个提交查询，那么它们将在调用线程中执行（目前所有索引都是这种情况）。 因此，让多个线程调用单个查询的搜索也是相对有效的。

但是，从多个线程调用批量查询是非常低效的，这将产生比核心更多的线程。

------

# 内部线程的性能（OpenML）

选择 OpenMP 线程的数量并不总是很明显。 有一些架构，其中设置的线程数少于核心数，从而显着提高了执行效率。 例如，在Intel E5-2680 v2上，将线程数设置为20而不是默认值40是很有用的。

使用 OpenBLAS BLAS 实现时，将线程策略设置为 PASSIVE 是很有用的：

```
export OMP_WAIT_POLICY=PASSIVE
```

------

# 异步搜索

与其他一些计算并行执行索引搜索操作可能很有用，包括：

- 单线程计算
- I/O 等待
- GPU 计算

这样，程序就可以并行运行。

对于Faiss CPU，与其他多线程计算（例如其他搜索）并行化是没有用的，因为这会产生太多线程并降低整体性能; 在递交给Faiss之前，应该将来自可能不同的用户线程的多个传入搜索排队并由用户聚合/批处理。

在多个GPU上并行运行操作当然是可行且有用的，其中每个CPU线程专用于在不同GPU上的内核启动，这就是IndexProxy和IndexShards的实现方式。

如何生成搜索线程：

- C++ 中， 使用 pthread_create + pthread_join
- Python 中， thread.start_new_thread + 一个锁， 或者使用 multiprocessing.dummy.Pool。 search、add、train 函数都会释放 GIL （Global Interpreter Lock）。