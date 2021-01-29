# FixMatch
This is an implementation of  ["FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence,"](https://arxiv.org/abs/2001.07685) NeurIPS'20. in pytorch. 

## Results: Error Rate (%)

We reproduced the experiments on CIFAR-10 with 40, 250, 4000 labeled data and 5000 validation samples as the official implementation of FixMatch. But due to the limitation of computational resources, we didn’t reproduce 5 "folds". Thus, our result based on 1 fold doesn’t have the standard deviation. Our model uses the Wide ResNet-28-2 with leaky ReLU activation function. Our results are comparable to the performance in the original paper.

|                       |                | CIFAR-10      |               |
| --------------------- | -------------- | ------------- | ------------- |
| Method                | 40 labels      | 250labels     | 4000 labels   |
| Official FixMatch(RA) | 13.81±3.37 | 5.07±0.65 | 4.26±0.05 |
| Ours(RA)              | 10.04          | 5.29          | 4.36          |



### Dependencies

```
pip install --upgrade git+https://github.com/pytorch/ignite
pip install -r requirements.txt
```



## Running

```
python run.py DATASET.label_num=250 DATASET.strongaugment='RA' 
```



## Checkpoint accuracy

```
python run_load_temp.py EXPERIMENT.resume_checkpoints='./checkpoints/'
```



## 



