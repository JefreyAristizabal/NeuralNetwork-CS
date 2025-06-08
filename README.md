# NeuralNetwork-CS

# 🧠 Neural Network from Scratch in C#

This project implements a **fully customizable feedforward neural network** from scratch in **C#**, without using any external machine learning libraries. It is designed to help you understand how neural networks work internally — including matrix operations, forward propagation, backpropagation, and activation functions.

## 🔧 Features

- Modular `NeuralNetwork` class for building multi-layer architectures  
- Support for multiple **activation functions** (e.g., Sigmoid, Tanh, ReLU)  
- **Matrix math implementation** from scratch (dot product, transpose, element-wise ops)  
- **Backpropagation** with gradient descent  
- Pluggable **loss functions** (currently MSE)  
- XOR training example included  
- Console-based training feedback (error over epochs)

## 🧪 Example: XOR Problem

```csharp
var net = new NeuralNetwork(2);                 // 2 input neurons
net.AddLayer(4, ActivationType.Sigmoid);        // Hidden layer with 4 neurons
net.AddLayer(1, ActivationType.Sigmoid);        // Output layer with 1 neuron

net.SetLearningRate(0.1);
net.SetLossFunction(LossType.MSE);

net.Train(Dataset.XOR_Inputs, Dataset.XOR_Outputs, epochs: 10000);
