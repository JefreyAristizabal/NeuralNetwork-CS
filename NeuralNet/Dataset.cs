using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public static class Dataset
    {
        public static double[][] XOR_Inputs = new double[][]
        {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };

        public static double[][] XOR_Outputs = new double[][]
        {
            new double[] { 0 },
            new double[] { 1 },
            new double[] { 1 },
            new double[] { 0 }
        };

        // Más adelante puedes agregar datasets como:
        // - Clasificación de puntos 2D
        // - Reconocimiento básico de dígitos (8x8)
        // - XOR con ruido
    }
}
