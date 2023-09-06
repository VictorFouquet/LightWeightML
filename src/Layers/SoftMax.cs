using MathNet.Numerics.LinearAlgebra;

namespace MachineLearning.Layers
{
    internal class SoftMax : ILayer
    {
        private Matrix<double> outputs;

        public SoftMax()
        {
            outputs = Matrix<double>.Build.Dense(1, 1);
        }

        public Matrix<double> Forward(Matrix<double> _inputs) {
            var tmp = _inputs.Map(n => Math.Exp(n));
            outputs = tmp.Divide(tmp.ToColumnArrays()[0].Sum());
            return outputs;
        }

        public Matrix<double> Backward(Matrix<double> outputGradient, double learningRate)
        {
            var outputsV = outputs.ToColumnArrays()[0];
            int n = outputsV.Length;
            double[] tmpV = new double[n * n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    tmpV[i * n + j] = outputsV[j];
            var tmp = Matrix<double>.Build.Dense(n, n, tmpV);
            return tmp.PointwiseMultiply(Matrix<double>.Build.DenseIdentity(n, n) - tmp.Transpose()).Multiply(outputGradient);
        }
    }
}
