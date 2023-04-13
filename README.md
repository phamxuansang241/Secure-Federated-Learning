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

## Usage
This section outlines the necessary steps for utilizing the Secure Federated Learning framework.

### Installation
```
$ git clone https://github.com/phamxuansang241/Secure-Federated-Learning.git
$ cd Secure-Federated-Learning
$ pip install -r requirements.txt
```

### Running the Project
Execute the following command to run the project:
```
$ python main.py -cf config.json
```
Here, config.json is a configuration file specifying the training settings, as well as client and server configurations.

### Configuration Template
```
Below is a sample configuration file template (config.json):
{
    "global_config": {
        "name": "Test (No noise)",
        "overwrite_experiment": true,
        "device": "cuda ",
        "training_mode": "dssgd",
        "compress_digit": 10,
        "dp_mode": false
    },
    "data_config": {
        "dataset_name": "smsspam",
        "data_sampling_technique": "iid"
    },
    "fed_config": {
        "global_epochs": 50,
        "local_epochs": 1,
        "nb_clients": 10,
        "fraction": 1.0,
        "batch_size": 64
    },
    "optim_config": {
        "lr": 0.001
    },
    "dp_config": {
        "epsilon": 10000,
        "delta": 1e-05,
        "clipping_norm": 1.0,
        "sigma": 0.1,
        "is_fixed_client_iter": false,
        "client_iter": 5000
    }
}
```
This configuration file consists of the following sections:

- global_config: Contains general settings for the experiment, such as device, training mode, and differential privacy mode.
- data_config: Specifies the dataset and data sampling technique to be used.
- fed_config: Contains parameters for federated learning, such as global and local epochs, number of clients, and batch size.
- optim_config: Provides optimization settings, such as the learning rate.
- dp_config: Configures differential privacy parameters, including epsilon, delta, clipping norm, and sigma.

Modify the parameters in the configuration file according to your specific requirements and experiment settings.
