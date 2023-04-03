# Secure Federated Learning Framework using Elgamal and Elliptic Curve Encryption

This repository contains a secure federated learning framework that uses Elgamal and Elliptic Curve encryption protocols for multi-party computation. The framework has been applied to training deep learning models in a distributed manner on CSIC2010, MNIST, and SMSSPAM datasets. 

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Background

Federated learning is a decentralized approach to train machine learning models in which data is stored locally on different devices or servers, and the models are trained on the data without transferring it to a centralized server. This approach provides a secure and privacy-preserving way of training machine learning models, as the data remains on the local devices and is not shared with a centralized server. 

However, federated learning still poses some security risks, such as malicious clients trying to manipulate the model or leak sensitive data. To mitigate these risks, this repository implements a secure federated learning framework using Elgamal and Elliptic Curve encryption protocols. These protocols provide strong security guarantees and ensure that the model is trained in a privacy-preserving and secure way.

## Installation

To install the required dependencies, run the following command:

