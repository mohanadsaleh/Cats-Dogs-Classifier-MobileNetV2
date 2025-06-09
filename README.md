# Cats-Dogs Classifier using MobileNetV2

This project implements a convolutional neural network (CNN) classifier to distinguish between cats and dogs images using transfer learning with the MobileNetV2 architecture.

## Project Overview

- Dataset: Cats and Dogs images organized in training and validation folders.
- Pretrained Model: MobileNetV2 (weights trained on ImageNet) without the top layer.
- Custom Classification Head with fully connected layers, dropout, and batch normalization.
- Trained for 20 epochs with Adam optimizer and a low learning rate.
- Input images resized to 224x224 pixels.
- Real-time data augmentation and normalization applied during training.

## Installation

Make sure you have the following packages installed:

```bash
pip install tensorflow matplotlib numpy
