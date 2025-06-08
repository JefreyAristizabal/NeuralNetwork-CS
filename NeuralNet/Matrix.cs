using System;

namespace NeuralNet
{
    public class Matrix
    {
        public int Rows { get; }
        public int Cols { get; }
        public double[,] Data;

        private static readonly Random Rand = new Random();

        public Matrix(int rows, int cols)
        {
            Rows = rows;
            Cols = cols;
            Data = new double[rows, cols];
            Randomize();
        }

        public void Randomize()
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Data[i, j] = Rand.NextDouble() * 2 - 1; // [-1, 1]
        }

        public static Matrix FromArray(double[] arr)
        {
            Matrix m = new Matrix(arr.Length, 1);
            for (int i = 0; i < arr.Length; i++)
                m.Data[i, 0] = arr[i];
            return m;
        }

        public double[] ToArray()
        {
            double[] arr = new double[Rows];
            for (int i = 0; i < Rows; i++)
                arr[i] = Data[i, 0];
            return arr;
        }

        public static Matrix Dot(Matrix a, Matrix b)
        {
            if (a.Cols != b.Rows)
                throw new Exception("Incompatible dimensions for dot product.");
            Matrix result = new Matrix(a.Rows, b.Cols);
            for (int i = 0; i < result.Rows; i++)
                for (int j = 0; j < result.Cols; j++)
                    for (int k = 0; k < a.Cols; k++)
                        result.Data[i, j] += a.Data[i, k] * b.Data[k, j];
            return result;
        }

        public void Add(Matrix other)
        {
            if (Rows != other.Rows || Cols != other.Cols)
                throw new Exception("Matrix dimensions must match for addition.");
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Data[i, j] += other.Data[i, j];
        }

        public void Add(double scalar)
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Data[i, j] += scalar;
        }

        public void Map(Func<double, double> func)
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Data[i, j] = func(Data[i, j]);
        }

        public static Matrix Map(Matrix m, Func<double, double> func)
        {
            Matrix result = new Matrix(m.Rows, m.Cols);
            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Cols; j++)
                    result.Data[i, j] = func(m.Data[i, j]);
            return result;
        }

        public void Multiply(Matrix m)
        {
            if (Rows != m.Rows || Cols != m.Cols)
                throw new Exception("Las dimensiones deben coincidir para multiplicación elemento a elemento.");

            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Data[i, j] *= m.Data[i, j];
        }

        // ✅ Método faltante para multiplicar por un escalar
        public void Multiply(double scalar)
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Data[i, j] *= scalar;
        }

        public static Matrix Transpose(Matrix m)
        {
            Matrix result = new Matrix(m.Cols, m.Rows);
            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Cols; j++)
                    result.Data[j, i] = m.Data[i, j];
            return result;
        }

        public static Matrix Subtract(Matrix a, Matrix b)
        {
            if (a.Rows != b.Rows || a.Cols != b.Cols)
                throw new Exception("Matrix dimensions must match for subtraction.");
            Matrix result = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    result.Data[i, j] = a.Data[i, j] - b.Data[i, j];
            return result;
        }

        public Matrix Copy()
        {
            Matrix copy = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    copy.Data[i, j] = Data[i, j];
            return copy;
        }
    }
}
