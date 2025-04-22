# Multiple Instance Learning: Attention to Instance Classification

This repository contains code for the paper *Multiple Instance Learning:
Attention to Instance Classification* from SPIE 2025 Medical Imaging.

## Description

The AMIL model is implemented as a PyTorch module in `amil.py`.
Besides that, we provide Lightning modules for AMIL (`amil_mnist.py`)
and our method (`ours_mnist.py`) to show an examle application on
MNIST (`mnist.py`). The Lightning modules employ LeNet5 (`lenet.py`).

Training & testing can be performed as:

    python amil_mnist.py --bagsize_mean 10 --bagsize_std 2 \
                         --train_bags 10 --seed 1

or

    python ours_mnist.py --bagsize_mean 10 --bagsize_std 2 \
                         --train_bags 10 --seed 1

The code was developed using torchvision/0.13.0 and PyTorch-Lightning/1.6.5.

## Citing

If you use our work in your research, please cite our paper:

    @inproceedings{barucic2025,
      author={Denis Baručić and Jan Kybic},
      title={Multiple instance learning: attention to instance classification},
      volume={13406},
      booktitle={Medical Imaging 2025: Image Processing},
      publisher={SPIE},
      pages={134060V},
      year={2025},
      doi={10.1117/12.3045059}
    }
