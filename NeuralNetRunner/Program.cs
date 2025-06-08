using NeuralNet;
using System;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("🔵 Entrenando red neuronal para XOR...");

        // Crear red neuronal: 2 entradas -> 4 neuronas ocultas -> 1 salida
        NeuralNetwork net = new NeuralNetwork(2);

        // ✅ Primera capa con inputSize
        net.AddLayer(2, 4, ActivationType.Sigmoid);
        // ✅ Capas siguientes sin inputSize
        net.AddLayer(1, ActivationType.Sigmoid);

        // Configurar red
        net.SetLearningRate(0.1);
        net.SetLossFunction(LossType.MSE);

        // Entrenar
        net.Train(Dataset.XOR_Inputs, Dataset.XOR_Outputs, epochs: 10000);

        // Probar
        Console.WriteLine("\n🧪 Resultados:");
        for (int i = 0; i < Dataset.XOR_Inputs.Length; i++)
        {
            double[] output = net.Predict(Dataset.XOR_Inputs[i]);
            Console.WriteLine($"Entrada: {Dataset.XOR_Inputs[i][0]}, {Dataset.XOR_Inputs[i][1]} => Salida: {output[0]:F4} (esperado: {Dataset.XOR_Outputs[i][0]})");
        }

        Console.WriteLine("\n✅ Entrenamiento completado.");
    }
}
