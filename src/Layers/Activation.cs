using MathNet.Numerics.LinearAlgebra;

namespace MachineLearning.Layers
{
    internal class Activation : ILayer
    {
        private readonly Func<double, double> activation;
        private readonly Func<double, double> derivative;
        private Matrix<double> outputs;

        public Activation(Func<double, double> _activation, Func<double, double> _derivative)
        {
            activation = _activation;
            derivative = _derivative;
            outputs = Matrix<double>.Build.Dense(1, 1);
        }

        public Matrix<double> Forward(Matrix<double> _inputs)
        {
            outputs = _inputs;
            return _inputs.Map(activation);
        }

        public Matrix<double> Backward(Matrix<double> outputGradient, double learningRate)
        {
            return outputGradient.PointwiseMultiply(outputs.Map(derivative));
        }
    }
}
