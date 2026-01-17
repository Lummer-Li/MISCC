<h2 align="center">✨Deep Multi-view Clustering with Intra-view Similarity and Cross-view Correlation Learning</h2>


<p align="center">
  <b>Pengyuan Li<sup>1</sup>, Dongxia Chang<sup>1</sup>, Yiming Wang<sup>2</sup>, Man Liu<sup>1</sup>, Zisen Kong<sup>1</sup>, Linhua Kong<sup>1</sup>, Yao Zhao<sup>1</sup></b>
</p>

<p align="center">
  <sup>1</sup>Institute of Information Science, Beijing Jiaotong University, Beijing, China<br>
  <sup>2</sup>School of Computer Science, Nanjing University of Posts and Telecommunications, Nanjing, China<br>
</p>

<p align="center">
  <!-- IEEE TMM Badge -->
  <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/IEEE TMM-2025-blueviolet.svg?style=flat-square" alt="IEEE TMM">
  </a>
  <!-- arXiv Badge -->
  <!-- <a href="https://arxiv.org/abs/2412.08345" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2412.08345-b31b1b.svg?style=flat-square" alt="arXiv Paper">
  </a> -->
  <!-- Contact Badge -->
  <a href="pengyuanli@bjtu.edu.cn" target="_blank">
    <img src="https://img.shields.io/badge/Email-pengyuanli%40bjtu.edu.cn-blue.svg" alt="Contact Author">
  </a>
</p>

<p align="center">
  🔥 Our work has been accepted by IEEE Transactions on Multimedia 2025!<br>
</p>

## Overview🔍
<div>
    <img src="https://github.com/Lummer-Li/MISCC/blob/main/MISCC.png" width="90%" height="90%">
</div>

**Figure 1. The framework of the proposed MISCC.**


**_Abstract -_** Deep multi-view clustering (MVC) has gained widespread attention as it can effectively mine consistent information from multiple views and improve clustering performance. However, view bias often exists between views (i.e., the quality differences between views). Treating all views equally inevitably destroys structural information when simply concatenating or summing the embedded representation of multiple views. To alleviate this issue, we propose a deep multi-view clustering with intra-view similarity and cross-view correlation learning (MISCC), facilitating the intra-view discriminability and inter-view complementarity. Specifically, we utilize the intra-view inherent structure information to dynamically identify semantically similar samples within each view. By aggregating their embedding representations, fine-grained structures are enhanced to boost intra-cluster compactness and inter-cluster separation. Then, we construct a cross-view correlation learning module to align semantically related views while preserving the distinctive features of irrelevant views. Based on them, a centralized clustering alignment strategy is proposed to align the similarity distribution and clustering structure between each view and the unified view, balancing the diverse information among multiple views. By jointly training these modules, the unified representation is optimized to capture more discriminative information from multiple views. Extensive experiments conducted on eleven multi-view datasets demonstrate that MISCC outperforms the state-of-the-art clustering methods.

## Datasets📚
We conduct extensive experiments on eleven widely-used multi-view datasets, including UCI-Digit, BBCSport, BBC4view, BDGP, HW2sources, Handwritten, LandUse-21, Mfeat, Webkb, STL10, and Cifar10 datasets. The details of these datasets are shown below.

| Dataset      | Samples  | Views  | View Dimensions         | Clusters  |
|--------------|----------|--------|-------------------------|-----------|
| BBCSport     | 544      | 2      | 3183/3203               | 5         |
| HW2sources   | 2000     | 2      | 784/256                 | 10         |
| BDGP         | 2500     | 2      | 1750/79                 | 5         |
| UCI-Digit    | 2000     | 3      | 216/76/64               | 10        |
| Webkb        | 203      | 3      | 1703/230/230            | 4        |
| BBC4view     | 685      | 4      | 4659/4633/4665/4684     | 5        |
| Handwritten  | 2000     | 6      | 216/76/64/6/240/47      | 10        |
| LandUse-21   | 2100     | 3      | 20/59/40                | 21        |
| Mfeat        | 2000     | 6      | 216/76/64/6/240/47      | 10        |
| STL10        | 13000    | 3      | 1024/512/2048           | 10         |
| Cifar10      | 50000    | 3      | 512/2048/1024           | 10        |




## Experimental Results🏆


**Table 1. Comparison with deep learning algorithms. The notation O/M symbol denotes an out-of-memory error.**
<div>
    <img src="https://github.com/Lummer-Li/MISCC/blob/main/assets/tab1.png" width="80%" height="96%">
</div>
<!-- <br> </br> -->

**Table 2. Ablation study on BBC4VIEW. In this Table, “✓” represents MISCC with the component.**
<div>
    <img src="https://github.com/Lummer-Li/MISCC/blob/main/assets/tab2.png" width="30%" height="96%">
</div>



## Getting Started🚀
### Data Preparation
The dataset should be organised as follows, taking BBCSport as an example:
```text
BBCSport
├── X
│   ├── X1
│   ├── X2
│   ├── ...
├── Y
```

### Training and Evaluation
- To train the MISCC, run: `main.py`. The prediction results obtained using the K-Means algorithm.



## Cite our work📝
```bibtex
@ARTICLE{li2026deep,
  author={Li, Pengyuan and Chang, Dongxia and Wang, Yiming and Liu, Man and Kong, Zisen and Kong, Linhua and Zhao, Yao},
  journal={IEEE Transactions on Multimedia}, 
  title={Deep Multi-View Clustering With Intra-View Similarity and Cross-View Correlation Learning}, 
  year={2026},
  volume={},
  number={},
  pages={1-11},
  keywords={Multi-view Clustering;Representation Learning;Deep Clustering},
  doi={10.1109/TMM.2026.3654390}}
```

## License📜
The source code is free for research and educational use only. Any commercial use should get formal permission first.



