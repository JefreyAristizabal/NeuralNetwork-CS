using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class NeuralNetwork
    {
        private List<Layer> layers = new();
        private double learningRate = 0.1;
        private LossType lossType = LossType.MSE;

        public NeuralNetwork(int inputSize)
        {
            // El inputSize se usa al crear explícitamente la primera capa
        }

        public void AddLayer(int outputSize, ActivationType activation)
        {
            if (layers.Count == 0)
                throw new Exception("Debe usar AddLayer(int inputSize, int outputSize, ActivationType) para la primera capa.");

            int inputSize = layers[^1].OutputSize;
            layers.Add(new Layer(inputSize, outputSize, activation));
        }

        public void AddLayer(int inputSize, int outputSize, ActivationType activation)
        {
            if (layers.Count == 0)
                layers.Add(new Layer(inputSize, outputSize, activation));
            else
                layers.Add(new Layer(layers[^1].OutputSize, outputSize, activation));
        }

        public double[] Predict(double[] inputArray)
        {
            Matrix input = Matrix.FromArray(inputArray);
            foreach (var layer in layers)
                input = layer.FeedForward(input);
            return input.ToArray();
        }

        public void Train(double[][] inputs, double[][] targets, int epochs)
        {
            for (int e = 0; e < epochs; e++)
            {
                double totalError = 0;

                for (int i = 0; i < inputs.Length; i++)
                {
                    Matrix input = Matrix.FromArray(inputs[i]);
                    Matrix output = FeedForward(input);
                    Matrix target = Matrix.FromArray(targets[i]);

                    // ✅ Convertimos a arreglos para MSE (double[]), que devuelve double
                    totalError += LossFunction.MSE(output.ToArray(), target.ToArray());

                    Backpropagate(target);
                }

                if (e % (epochs / 10) == 0 || e == epochs - 1)
                {
                    Console.WriteLine($"Epoch {e + 1}/{epochs} – Error: {totalError / inputs.Length:F6}");
                }
            }
        }

        private Matrix FeedForward(Matrix input)
        {
            foreach (var layer in layers)
                input = layer.FeedForward(input);
            return input;
        }

        private void Backpropagate(Matrix target)
        {
            Matrix error = Matrix.Subtract(target, layers[^1].Outputs);

            for (int i = layers.Count - 1; i >= 0; i--)
            {
                Layer layer = layers[i];

                Matrix gradient = layer.GetActivationDerivative();
                gradient.Multiply(error); // Element-wise: derivada * error
                gradient.Multiply(learningRate);

                Matrix prevOutputT = i == 0
                    ? Matrix.Transpose(layer.Inputs)
                    : Matrix.Transpose(layers[i - 1].Outputs);

                Matrix delta = Matrix.Dot(gradient, prevOutputT);

                layer.Weights.Add(delta);
                layer.Biases.Add(gradient);

                if (i != 0)
                {
                    Matrix weightsT = Matrix.Transpose(layer.Weights);
                    error = Matrix.Dot(weightsT, error);
                }
            }
        }

        public void SetLearningRate(double lr) => learningRate = lr;
        public void SetLossFunction(LossType type) => lossType = type;
    }
}
