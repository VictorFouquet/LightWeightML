
using MachineLearning.Enums;
using MachineLearning.Layers;

namespace MachineLearning.Builders
{
    internal static class ActivationBuilder
    {
        public static ILayer Get(ActivationEnum layerType)
        {

            switch (layerType)
            {
                case ActivationEnum.Linear:
                    return new Activation(ActivationFunction.Linear, ActivationFunction.LinearDerivative);
                case ActivationEnum.TanH:
                    return new Activation(ActivationFunction.TanH, ActivationFunction.TanHDerivative);
                case ActivationEnum.Relu:
                    return new Activation(ActivationFunction.Relu, ActivationFunction.ReluDerivative);
                case ActivationEnum.Sigmoid:
                    return new Activation(ActivationFunction.Sigmoid, ActivationFunction.SigmoidDerivative);
                default:
                    return new Activation(ActivationFunction.Linear, ActivationFunction.LinearDerivative);
            }
        }
    }
}
