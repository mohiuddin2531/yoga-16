# Integrating Skeleton-Based Representations for Robust Yoga Pose Classification

**Journal Title**: Integrating Skeleton-Based Representations for Robust Yoga Pose Classification: A Comparative Analysis of Deep Learning Models

**Authors**: 
- Mohammed Mohiuddin, B.Sc (First Author)
- Syed Mohammod Minhaz Hossain
- Sumaiya Khanam, B.Sc
- Prionkar Barua, B.Sc
- Aparup Barua, B.Sc
- MD Tamim Hossain, B.Sc


**Affiliation**: Premier University

This repository accompanies the research work titled "Integrating Skeleton-Based Representations for Robust Yoga Pose Classification: A Comparative Analysis of Deep Learning Models." The goal of this project is to explore and compare various deep learning architectures to effectively classify yoga poses using skeleton-based representations.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Datasets](#datasets)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Yoga pose classification is a challenging problem, especially when it comes to achieving robustness in real-world scenarios. This repository implements skeleton-based methods to extract meaningful features and compares the performance of multiple deep learning models, including CNNs, RNNs, and Transformers.

The repository includes code, datasets, and results analysis for reproducibility and further exploration.

## Features

- Implementation of skeleton-based preprocessing techniques.
- Comparative analysis of multiple deep learning models:
  - Convolutional Neural Networks (CNNs)
  - Recurrent Neural Networks (RNNs)
  - Transformer-based models
- Performance metrics such as accuracy, precision, recall, and F1-score.
- Visualization of results and misclassified poses.

## Datasets

The repository supports public yoga pose datasets. If you plan to use your custom dataset, ensure the following format:

- **Input:** Skeleton keypoints in JSON or CSV format.
- **Labels:** Pose categories as integers or class names.

Preprocessing scripts are available in the `preprocessing` folder, including YOLOv8 and MediaPipe-based preprocessing codes.

## Model Architectures

The following models are included in the repository:

1. **VGG16**: Modified for pose classification.
2. **ResNet50**: Known for its deep residual connections.
3. **Xception**: Efficient architecture for feature extraction.

Each model's implementation can be found in the `models` folder as `.ipynb` files.

## Results

The results of the comparative analysis are documented in the `results` folder. Key highlights include:

- VGG16 achieved the highest accuracy for static poses.
- ResNet50 provided a good balance between accuracy and computational efficiency.
- Xception excelled in scenarios requiring high feature extraction capabilities.

Refer to the paper for detailed performance metrics.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/yoga-pose-classification.git
   cd yoga-pose-classification
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download and preprocess the dataset as instructed in the `datasets` folder.

## Usage

1. Train a model:

   ```bash
   python train.py --model VGG16 --epochs 50 --dataset ./data/yoga_dataset
   ```

2. Evaluate a trained model:

   ```bash
   python evaluate.py --model VGG16 --weights ./checkpoints/vgg16_best.pth --dataset ./data/yoga_dataset
   ```

3. Visualize results:

   ```bash
   python visualize.py --results ./results/vgg16_results.json
   ```

## Contributing

Contributions are welcome! If you'd like to contribute, please:

- Fork the repository.
- Create a new branch.
- Submit a pull request with detailed explanations.

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more details.
