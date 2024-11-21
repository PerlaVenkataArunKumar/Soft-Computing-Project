# Traffic Sign Recognition 🚦

## Project Overview
This project implements a Traffic Sign Recognition System using a combination of Convolutional Neural Networks (CNNs) and Genetic Algorithms (GAs). The system is designed to classify 43 types of traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

By leveraging CNN for image classification and GA for hyperparameter optimization, the project achieves high accuracy with minimal manual tuning.

## Objectives
Design a CNN to classify 43 types of traffic signs.
Use a Genetic Algorithm (GA) to optimize hyperparameters for enhanced performance.

## Methodology
1. Convolutional Neural Network (CNN)

## CNN Architecture:
Input Layer: Processes 30x30x3 RGB images.

## Convolutional Layers:
Layer 1: 64 filters, kernel size (5, 5), activation: ReLU.
Layer 2: 128 filters, kernel size optimized by GA, activation: ReLU.

## Fully Connected Layers:
Dense Layer: 128 neurons, activation: ReLU.
Output Layer: 43 neurons, activation: Softmax.
Optimizer: Adam.
Loss Function: Sparse Categorical Crossentropy.
2. Genetic Algorithm (GA)

## GA Process:
Representation: Individuals represent hyperparameters:
[filters_1, filters_2, kernel_size, learning_rate, batch_size].
Fitness Evaluation: Validation accuracy of CNN trained with individual hyperparameters.
Selection: Fittest individuals are selected for reproduction.
Crossover: Combines parent hyperparameters to produce offspring.
Mutation: Introduces diversity by altering hyperparameters randomly.
Iteration: Process repeats for multiple generations to enhance fitness.

## Implementation:
### Integration of CNN and GA:
CNN is built and trained using hyperparameters from GA individuals.
Validation accuracy serves as the fitness score.

## Final Model Training:
Best hyperparameters from GA are used to train the final CNN model over 15 epochs.

## Results
### Best Hyperparameters:

Filters (Layer 1): 64
Filters (Layer 2): 128
Kernel Size: (3, 3)
Learning Rate: 0.001
Batch Size: 32
Performance Metrics:
Validation Accuracy: 96%
Test Accuracy: 94%
## Conclusion:
Combining CNN with Genetic Algorithms for hyperparameter optimization results in:

Efficient exploration of hyperparameter space.
Enhanced model performance without manual tuning.
Key Takeaways:
Genetic Algorithms can automate and enhance deep learning workflows.
Traffic Sign Classification is a critical task in autonomous driving systems.
Dataset
German Traffic Sign Recognition Benchmark (GTSRB)
Training: 80% of the dataset.
Testing: 20% of the dataset.
