
using MachineLearning.Enums;
using MachineLearning.Layers;

namespace MachineLearning.Builders
{
    internal static class LayerBuilder
    {
        public static ILayer Get(LayerEnum layerType, int numNodesIn = 0, int numNodesOut = 0)
        {
            switch (layerType)
            {
                case LayerEnum.Dense:
                    return new Dense(numNodesIn, numNodesOut);
                case LayerEnum.SoftMax:
                    return new SoftMax();
                case LayerEnum.Linear:
                    return ActivationBuilder.Get(ActivationEnum.Linear);
                case LayerEnum.Relu:
                    return ActivationBuilder.Get(ActivationEnum.Relu);
                case LayerEnum.Sigmoid:
                    return ActivationBuilder.Get(ActivationEnum.Sigmoid);
                case LayerEnum.TanH:
                    return ActivationBuilder.Get(ActivationEnum.TanH);
                default:
                    return ActivationBuilder.Get(ActivationEnum.Linear);
            }
        }
    }
}
