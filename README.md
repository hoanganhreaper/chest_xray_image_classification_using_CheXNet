# CheXNet for Pneumonia Classification

## Overview
This project implements a pre-trained DenseNet-121 model for classifying Chest X-Ray images to detect pneumonia. The model is fine-tuned to fit the specific dataset of chest X-ray images.

## Dataset
The dataset used for training and evaluating the model consists of chest X-ray images categorized into classes. Each image is labeled as either showing signs of pneumonia or being normal.

## Requirements
- Python 3.x
- PyTorch
- TorchVision
- NumPy
- Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hoanganhreaper/chest_xray_image_classification_using_CheXNet.git
   cd your-repository
2. Install the required packages:
   pip install -r requirements.txt

## Usage
Download the pre-trained DenseNet-121 model weights and place them in the appropriate directory:
   wget https://download.pytorch.org/models/densenet121-a639ec97.pth -P /path/to/weights/

## Model Architecture
The model architecture is based on DenseNet-121. The classifier layer is modified to match the number of classes in the dataset.

## Fine-Tuning
The pre-trained DenseNet-121 model is fine-tuned by:
- Freezing the convolutional layers to prevent them from being updated during training.
- Replacing the final fully connected layer with a new layer that matches the number of classes in the dataset.

## Results
The model achieves the results over 99% in train dataset and 88% in test dataset
