using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;


namespace MachineLearning
{
    /// <summary>
    /// Tiny <c>NeuralNetwork</c> class to create and train multi-layer networks then make predictions. 
    /// </summary>
    public class NeuralNetwork
    {
        public ILayer[] layers;

        /// <summary>
        /// Note that the input size of any layer having a layer before itself should be the same
        /// as this previous layer's output size.
        /// </summary>
        /// <param name="_layers"></param>
        public NeuralNetwork(params ILayer[] _layers)
        {
            layers = _layers;
        }

        /// <summary>
        /// Performs training by propagating data forward to make a prediction.
        /// Computes cost of the prediction with mean squared error.
        /// Backpropagates the error to adjust the layers weights and biases.
        /// </summary>
        /// <param name="dataSet"></param>
        /// <param name="epochs"></param>
        /// <param name="learningRate"></param>
        public void Train(DataPoint[] dataSet, int epochs, double learningRate)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var dataPoint in dataSet)
                {
                    var outputs = Matrix<double>.Build.Dense(dataPoint.X.Length, 1, dataPoint.X);
                    var expected = Matrix<double>.Build.Dense(dataPoint.Y.Length, 1, dataPoint.Y);
                    foreach (var layer in layers)
                        outputs = layer.Forward(outputs);

                    var mse = expected.Subtract(outputs).Map(n => n * n).ToColumnArrays()[0].Mean();

                    var grad = expected.Subtract(outputs).Divide(dataPoint.Y.Length).Multiply(2);
                    for (int i = layers.Length - 1; i >= 0; i--)
                    {
                        grad = layers[i].Backward(grad, learningRate);
                    }
                }
            }
        }

        /// <summary>
        /// Makes a prediction for a given point of data.
        /// Should be trained first.
        /// </summary>
        /// <param name="X"></param>
        /// <returns></returns>
        public double[] Predict(double[] X)
        {
            var outputs = Matrix<double>.Build.Dense(X.Length, 1, X);
            foreach (var layer in layers)
                outputs = layer.Forward(outputs);
            return outputs.ToColumnArrays()[0];
        }
    }
}