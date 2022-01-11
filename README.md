# MUGE Text To Image Generation Baseline

## Requirements and Installation
More details see [fairseq](https://github.com/pytorch/fairseq). Briefly,

* python == 3.6.4
* pytorch == 1.7.1

1. **Installing fairseq and other requirements**  
```bash
git clone https://github.com/MUGE-2021/image-caption-baseline
cd muge_baseline/
pip install -r requirements.txt
cd fairseq/
pip install --editable .
```

2. **Downloading data and place to `dataset/` directory,**
    file structure is 
```markdown
text2image-baseline
    - dataset
        - ECommerce-T2I
            - T2I_train.img.tsv
            - T2I_train.text.tsv
            - ...
``` 

## Getting Started
The model is a BART-like model with vqgan as a image tokenizer, please see `models/t2i_baseline.py` for detailed model structure.
### Training 
```bash
cd run_scripts/; bash train_t2i_vqgan.sh
```
Model training takes about 5 hours.

### Inference
```bash
cd run_scripts/; bash generate_t2i_vqgan.sh
```
See results in `results/` directory.

## Reference
```
@inproceedings{M6,
  author    = {Junyang Lin and
               Rui Men and
               An Yang and
               Chang Zhou and
               Ming Ding and
               Yichang Zhang and
               Peng Wang and
               Ang Wang and
               Le Jiang and
               Xianyan Jia and
               Jie Zhang and
               Jianwei Zhang and
               Xu Zou and
               Zhikang Li and
               Xiaodong Deng and
               Jie Liu and
               Jinbao Xue and
               Huiling Zhou and
               Jianxin Ma and
               Jin Yu and
               Yong Li and
               Wei Lin and
               Jingren Zhou and
               Jie Tang and
               Hongxia Yang},
  title     = {{M6:} {A} Chinese Multimodal Pretrainer},
  year      = {2021},
  booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining},
  pages     = {3251–3261},
  numpages  = {11},
  location  = {Virtual Event, Singapore},
}

@article{M6-T,
  author    = {An Yang and
               Junyang Lin and
               Rui Men and
               Chang Zhou and
               Le Jiang and
               Xianyan Jia and
               Ang Wang and
               Jie Zhang and
               Jiamang Wang and
               Yong Li and
               Di Zhang and
               Wei Lin and
               Lin Qu and
               Jingren Zhou and
               Hongxia Yang},
  title     = {{M6-T:} Exploring Sparse Expert Models and Beyond},
  journal   = {CoRR},
  volume    = {abs/2105.15082},
  year      = {2021}
}
```

