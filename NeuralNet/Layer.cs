using System;

namespace NeuralNet
{
    public class Layer
    {
        public int InputSize, OutputSize;
        public Matrix Weights;
        public Matrix Biases;
        public Matrix Outputs;
        public Matrix Inputs;
        public Matrix Gradients;
        public ActivationType Activation;

        public Layer(int inputSize, int outputSize, ActivationType activation)
        {
            InputSize = inputSize;
            OutputSize = outputSize;
            Activation = activation;

            Weights = new Matrix(outputSize, inputSize);
            Biases = new Matrix(outputSize, 1);
        }

        public Matrix FeedForward(Matrix input)
        {
            Inputs = input;
            Outputs = Matrix.Dot(Weights, input);
            Outputs.Add(Biases);
            Outputs.Map(GetActivation(Activation)); // Map es un método de instancia
            return Outputs;
        }

        public Matrix GetActivationDerivative()
        {
            // Creamos una copia de Outputs para no modificar el original
            Matrix derivative = Outputs.Copy();
            derivative.Map(GetActivationDerivative(Activation)); // Map es instancia
            return derivative;
        }

        private Func<double, double> GetActivation(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.Sigmoid:
                    return ActivationFunction.Sigmoid;
                case ActivationType.Tanh:
                    return ActivationFunction.Tanh;
                case ActivationType.ReLU:
                    return ActivationFunction.ReLU;
                case ActivationType.LeakyReLU:
                    return ActivationFunction.LeakyReLU;
                default:
                    throw new NotImplementedException();
            }
        }

        private Func<double, double> GetActivationDerivative(ActivationType type)
        {
            switch (type)
            {
                case ActivationType.Sigmoid:
                    return ActivationFunction.SigmoidDerivative;
                case ActivationType.Tanh:
                    return ActivationFunction.TanhDerivative;
                case ActivationType.ReLU:
                    return ActivationFunction.ReLUDerivative;
                case ActivationType.LeakyReLU:
                    return ActivationFunction.LeakyReLUDerivative;
                default:
                    throw new NotImplementedException();
            }
        }
    }
}
