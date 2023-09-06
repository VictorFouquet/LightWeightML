using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace MachineLearning.Layers
{
    internal class Dense : ILayer
    {
        public int numNodesIn, numNodesOut;

        public Matrix<double> weights;
        public Matrix<double> bias;
        public Matrix<double> inputs;
        public Matrix<double> outputs;

        public Dense(int _numNodesIn, int _numNodesOut)
        {
            numNodesIn = _numNodesIn;
            numNodesOut = _numNodesOut;

            inputs = Matrix<double>.Build.Dense(_numNodesIn, 1);
            outputs = Matrix<double>.Build.Dense(_numNodesOut, 1);

            var gaussian = new Normal();
            var randBias = gaussian.Samples().Take(_numNodesOut).ToArray();
            var randWeights = gaussian.Samples().Take(_numNodesIn * _numNodesOut).ToArray();
            bias = Matrix<double>.Build.Dense(_numNodesOut, 1, randBias);
            weights = Matrix<double>.Build.Dense(_numNodesOut, _numNodesIn, randWeights);
        }

        public Matrix<double> Forward(Matrix<double> _inputs)
        {
            inputs = _inputs;
            weights.Multiply(inputs, outputs);
            outputs = outputs.Add(bias);

            return outputs.Clone();
        }

        public Matrix<double> Backward(Matrix<double> outputGradient, double learningRate)
        {

            var weightsGradients = outputGradient.Multiply(inputs.Transpose());

            weights += learningRate * weightsGradients;
            bias += learningRate * outputGradient;

            var inputGradients = weights.Transpose().Multiply(outputGradient);

            return inputGradients;
        }
    }
}
