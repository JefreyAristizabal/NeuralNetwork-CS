using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class ActivationFunction
    {
        public static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
        public static double SigmoidDerivative(double y) => y * (1 - y); // y = sigmoid(x)

        public static double Tanh(double x) => Math.Tanh(x);
        public static double TanhDerivative(double y) => 1 - y * y;

        public static double ReLU(double x) => x > 0 ? x : 0;
        public static double ReLUDerivative(double y) => y > 0 ? 1 : 0;

        public static double LeakyReLU(double x) => x > 0 ? x : 0.01 * x;
        public static double LeakyReLUDerivative(double y) => y > 0 ? 1 : 0.01;

        public static double Softmax(double x) => throw new NotImplementedException("Handled per layer");

        // You can expand with more as needed.
    }

    public enum ActivationType
    {
        Sigmoid,
        Tanh,
        ReLU,
        LeakyReLU
    }
}
