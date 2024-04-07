# CertifiedSNN
Code for the paper: **Certified Adversarial Robustness for Rate Encoded Spiking Neural Networks**


**Citation:**
@inproceedings{
mukhoty2024certified,
title={Certified Adversarial Robustness for Rate Encoded Spiking Neural Networks},
author={Bhaskar Mukhoty and Hilal AlQuabeh and Giulia De Masi and Huan Xiong and Bin Gu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=5bNYf0CqxY}
}

**Requirements:**

    pytorch 1.11.0
    torchvision 0.15.2
    torchattacks 3.4.0
    statsmodels 0.14.0
    scipy 1.10.1
    
**Training:**
The below command will train a VGG11 model on the CIFAR-10 dataset using rate encoding with no attack, and a latency, T=4.
python main_train.py


**Testing:**
To test a trained model, use the following command:
python main_test.py --id "name_of_model"

For example:
python main_test.py --id vgg11_rate_T4_clean

If an attack is applicable, you can specify it using the --attack flag with one of the following options: "pgd", "fgsm", "gn", or "pgd-l1".

**Certification:**
To certify a trained model using default statistical testing parameters, run:
python certify.py
The default statistical testing parameters include:
    m0 = 10
    m = 100
    error_rate = 1

**Contact:**
Bhaskar Mukhoty ({firstname}.{lastname}@gmail.com)
Hilal AlQuabeh (h{lastname}@gmail.com)
