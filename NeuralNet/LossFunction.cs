using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class LossFunction
    {
        public static double MSE(double[] outputs, double[] targets)
        {
            double sum = 0;
            for (int i = 0; i < outputs.Length; i++)
                sum += Math.Pow(targets[i] - outputs[i], 2);
            return sum / outputs.Length;
        }

        public static double[] MSE_Derivative(double[] outputs, double[] targets)
        {
            double[] result = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
                result[i] = outputs[i] - targets[i];
            return result;
        }

        // Más adelante podemos añadir CrossEntropy, Huber, etc.
    }

    public enum LossType
    {
        MSE
    }
}
