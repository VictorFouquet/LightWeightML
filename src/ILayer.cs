using MathNet.Numerics.LinearAlgebra;

namespace MachineLearning
{
    public interface ILayer
    {
        public Matrix<double> Forward(Matrix<double> inputs);
        public Matrix<double> Backward(Matrix<double> costGradient, double learningRate);
    }
}
