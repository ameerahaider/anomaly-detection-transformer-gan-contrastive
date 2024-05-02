# Transformer Autoencoder with Generative Adversarial Network (GAN) and Contrastive Learning for Anomaly Detection

This repository contains a PyTorch implementation of a Transformer Autoencoder combined with Generative Adversarial Network (GAN) and Contrastive Learning techniques for anomaly detection. The model is trained on the Pima Indians Diabetes dataset.

## Contents

- `transformer_autoencoder_gan_contrastive.ipynb`: Jupyter Notebook containing the implementation.
- `README.md`: This README file providing an overview of the project.

## Dependencies

- Python
- PyTorch
- Pandas
- Matplotlib

## Getting Started

1. Clone this repository:

```
git clone https://github.com/your_username/transformer-autoencoder-gan-contrastive.git
```

2. Install the required dependencies using pip:

```
pip install torch pandas matplotlib
```

3. Open and run the `transformer_autoencoder_gan_contrastive.ipynb` notebook using Jupyter or any compatible environment.

## Overview

- The dataset used in this project is the Pima Indians Diabetes dataset, containing various health-related features.
- The code begins with data preprocessing steps such as normalization, handling missing values, and removing duplicates.
- It then defines a Transformer Autoencoder model for feature extraction and reconstruction.
- Next, a Generator and a Discriminator are defined for adversarial training.
- Contrastive Learning is applied to the autoencoder for anomaly detection.
- Training loops for the Generator, Discriminator, and Autoencoder are implemented.
- Evaluation metrics such as MSE (Mean Squared Error) are calculated for anomaly detection.
- Anomalies are detected based on the predefined threshold.

## Usage

- Execute the code cells in the provided notebook sequentially.
- Adjust hyperparameters, model architecture, and training parameters as needed.
- Monitor training progress and evaluate model performance using provided metrics.

## References

- [Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- PyTorch Documentation: [torch.nn](https://pytorch.org/docs/stable/nn.html), [torch.optim](https://pytorch.org/docs/stable/optim.html), [torch.utils.data](https://pytorch.org/docs/stable/data.html)
- [Generative Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661)
- [Contrastive Learning](https://arxiv.org/abs/2002.05709)
