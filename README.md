# MER 

- Seed
  - random.seed(0)
  - torch.manual_seed(0)
  - np.random.seed(0)


## **GOAL MACRO F1 0.8676**
MER with casme2
with MACRO F1 suggested in https://facial-micro-expressiongc.github.io/MEGC2019/images/MEGC2019%20Recognition%20Challenge.pdf 
_________________________________________

- data_aug_nodenoise.ipynb holds image augmentation pipeline
- Link to download models: https://tm-mmnet-eval.s3.ap-southeast-1.amazonaws.com/V3_eval.zip
  - Run evaluation with f1calc.ipynb after zip extraction to evaluate

Models based on MMnet
@article{li2022mmnet,
  title={MMNet: Muscle motion-guided network for micro-expression recognition},
  author={Li, Hanting and Sui, Mingzhe and Zhu, Zhaoqing and Zhao, Feng},
  journal={arXiv preprint arXiv:2201.05297},
  year={2022}
}
