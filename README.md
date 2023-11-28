# Adversarial Preference Optimization

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/Linear95/APO/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/Linear95/APO/blob/main/DATA_LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

This repo contains the implementation of [Adversarial Preference Optimization](https://arxiv.org/abs/2311.08045) (APO). 

We let the reward model (RM) and LLM agent play a min-max game, through which both models can be further enhanced without additional preference annotation.

<p align="center">
  <img src="figures/apo_framework_shot.png" height="75%" width="75%">
</p>

Currently, the repo contains:
- [Split Helpful\&Harmless](data/hh-split) (HH) dataset
- [GPT-4 responses](data/hh-split/rm_data/hh_split_rm.golden.json) as golden annotation on HH-RM training set

We are continuously updating this repo for the reproduction of APO experiments.

## Data \& Annotation

To seperately update RM and LLM, we split the cleaned [Helpful\&Harmless](https://github.com/Linear95/DSP/tree/main/data) (HH) dataset into a RM training set and a LLM training set.

| Data Type| HH-RM Train Set | HH-LLM Train Set| HH Test Set|
| --------| ----------| -------| --------|
| Preference Pairs | [RM training set](data/hh-split/rm_data/hh_split_rm.train.json) | [RM validation set](data/hh-split/eval_data/hh_split_llm.valid.json) | [RM testing set](data/hh-split/eval_data/hh_cleaned_origin.test.json)|
| Golden Answers | [APO positive responses](data/hh-split/rm_data/hh_split_rm.golden.json) | - | -|
|User Queries | [APO negative responses](data/hh-split/rm_data/hh_split_rm_alpaca_v0.sample.json) (Alpaca samples)| [LLM (Alpaca) rejection samples](data/hh-split/llm_data/hh_split_llm_alpaca_v0.sample.json)| [LLM testing Queries](data/hh-split/eval_data/hh_cleaned_origin.test.json)|



## Citation
```
@article{cheng2023adversarial,
  title={Adversarial Preference Optimization},
  author={Cheng, Pengyu and Yang, Yifan and Li, Jian and Dai, Yong and Du, Nan},
  journal={arXiv preprint arXiv:2311.08045},
  year={2023}
}
```
