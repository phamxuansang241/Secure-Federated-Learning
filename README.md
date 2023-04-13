# Secure Federated Learning Framework with Encryption Aggregation and Integer Encoding Method.

This research repository presents an advanced, secure federated learning framework, incorporating Elgamal and Elliptic Curve encryption protocols to facilitate multi-party computation among distributed clients. Additionally, the project employs an integer encoding technique to optimize client model weights, substantially reducing both computational and communication costs.

The proposed framework has been effectively applied to address complex challenges in a distributed data setting, including web attack detection, handwritten digit recognition, and spam email identification. Specifically, the following applications were explored:

- Web attack detection: Investigating and analyzing HTTP request data to identify potential threats. Experiments were conducted utilizing the CSIC2010 dataset.
- Handwritten digit recognition: Accurately recognizing handwritten numerical characters. Experiments were conducted using the MNIST dataset.
- Spam email detection: Identifying and filtering unsolicited emails. Experiments were conducted employing the SMSSPAM dataset.

## Table of Contents

- [Background](#background)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Background

Federated learning is a decentralized approach to train machine learning models in which data is stored locally on different devices or servers, and the models are trained on the data without transferring it to a centralized server. This approach provides a secure and privacy-preserving way of training machine learning models, as the data remains on the local devices and is not shared with a centralized server. 

However, federated learning still poses some security risks, such as malicious clients trying to manipulate the model or leak sensitive data. To mitigate these risks, this repository implements a secure federated learning framework using Elgamal and Elliptic Curve encryption protocols. These protocols provide strong security guarantees and ensure that the model is trained in a privacy-preserving and secure way.

## Requirements

To install the required dependencies in requirements.txt:
- functorch==0.2.1
- gmpy2==2.1.5
- opacus==1.3.0
- torch==1.12.1
- torchvision==0.13.1
- torchtext==0.13.1
- tinyec==0.4.
- seaborn==0.11.20
- pandas==1.5.2
- scikit-learn==1.2.0
- numpy==1.24.1
- matplotlib==3.7.1

