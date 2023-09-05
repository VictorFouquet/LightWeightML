using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace MachineLearning
{
    /// <summary>
    /// Layer used to perform forward and backward propagation in a neural network
    /// </summary>
    public class Layer
    {
        public int numNodesIn, numNodesOut;

        public Matrix<double> weights;
        public Matrix<double> bias;
        public Matrix<double> inputs;
        public Matrix<double> outputs;
        public ActivationEnum activation;

        public Layer(int _numNodesIn, int _numNodesOut, ActivationEnum _activation)
        {
            numNodesIn = _numNodesIn;
            numNodesOut = _numNodesOut;
            activation = _activation;

            inputs = Matrix<double>.Build.Dense(_numNodesIn, 1);
            outputs = Matrix<double>.Build.Dense(_numNodesOut, 1);

            // Inititializes weights and biases
            var gaussian = new Normal(0.0, 0.3);
            var randBias = gaussian.Samples().Take(_numNodesOut).ToArray();
            var randWeights = gaussian.Samples().Take(_numNodesIn * _numNodesOut).ToArray();
            bias = Matrix<double>.Build.Dense(_numNodesOut, 1, randBias);
            weights = Matrix<double>.Build.Dense(_numNodesOut, _numNodesIn, randWeights);
        }

        /// <summary>
        /// Propagates input data through the weights, adds biases and maps the output with activation function
        /// </summary>
        /// <param name="_inputs"></param>
        /// <returns>The activated value of the outputs node</returns>
        public Matrix<double> Forward(Matrix<double> _inputs)
        {
            // Performs a dot product between the layer weights and the given inputs, stored in outputs
            inputs = _inputs;
            weights.Multiply(inputs, outputs);
            // Adds bias to the output
            outputs = outputs.Add(bias);
            // Returns the outputs mapped with activation function
            return Activate();
        }

        /// <summary>
        /// Performs the backward propagation of the error
        /// </summary>
        /// <param name="costGradient"></param>
        /// <param name="learningRate"></param>
        /// <returns>The gradient of the input nodes with respect to the provided error</returns>
        public Matrix<double> Backward(Matrix<double> costGradient, double learningRate)
        {
            var weightsGradients = Matrix<double>.Build.Dense(numNodesOut, numNodesIn);
            
            // Computes output gradient - Hadamard mul
            var outputGradients = costGradient.PointwiseMultiply(ActivateDerivative());

            // Computes weights gradients by performing dot product between output gradients and inputs transposed
            outputGradients.Multiply(inputs.Transpose(), weightsGradients);

            // Updates weights and biases according to their respective gradients
            weights += learningRate * weightsGradients;
            bias += learningRate * outputGradients;

            // Computes inputs gradients by performin dot product between weights transposed and ouput gradients
            var inputGradients = weights.Transpose().Multiply(outputGradients);

            return inputGradients;
        }

        /// <summary>
        /// Maps the layer's output with its activation function
        /// </summary>
        /// <returns>Activated output data</returns>
        private Matrix<double> Activate()
        {
            switch (activation)
            {
                case ActivationEnum.TanH:
                    return outputs.Map(Activation.TanH);
                case ActivationEnum.Relu:
                    return outputs.Map(Activation.Relu);
                case ActivationEnum.Sigmoid:
                    return outputs.Map(Activation.Sigmoid);
                default:
                    return Matrix<double>.Build.Dense(numNodesOut, 1);
            }
        }

        /// <summary>
        /// Maps the layer's output its activation function derivative
        /// </summary>
        /// <returns>Gradient of the output</returns>
        private Matrix<double> ActivateDerivative()
        {
            switch (activation)
            {
                case ActivationEnum.TanH:
                    return outputs.Map(Activation.TanHDerivative);
                case ActivationEnum.Relu:
                    return outputs.Map(Activation.ReluDerivative);
                case ActivationEnum.Sigmoid:
                    return outputs.Map(Activation.SigmoidDerivative);
                default:
                    return Matrix<double>.Build.Dense(numNodesIn, 1);
            }
        }
    }
}
