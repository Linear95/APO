# Adversarial Preference Optimization

Collecting human feedback to update the reward model (RM) in RLHF can be practically exhausting. Is there any efficient way to automatically generate preference comparisons for RM? We proposed [Adversarial Preference Optimization](https://arxiv.org/abs/2311.08045) (APO), where we let RM and LLM play a min-max game. Through APO, both RM and LLM can be further enhanced without additional preference annotation.

<p align="center">
  <img src="figures/apo_framework_shot.png" height="60%" width="60%">
</p>

Currently, this repo contains:
- the split Helpful\&Harmless dataset
- the GPT-4 responses on HH-RM set as golden data

We are continuously updating this repo for the reproduction of APO experiments.


## Citation
```
@article{cheng2023adversarial,
  title={Adversarial Preference Optimization},
  author={Cheng, Pengyu and Yang, Yifan and Li, Jian and Dai, Yong and Du, Nan},
  journal={arXiv preprint arXiv:2311.08045},
  year={2023}
}
```
