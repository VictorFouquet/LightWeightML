using MachineLearning.Builders;
using MachineLearning.Enums;

namespace MachineLearning
{
    /// <summary>
    /// Layer used to perform forward and backward propagation in a neural network
    /// </summary>
    public static class Layer
    {
        public static ILayer Dense(int numInputs, int numOutputs)
        {
            return LayerBuilder.Get(LayerEnum.Dense, numInputs, numOutputs);
        }

        public static ILayer Linear()
        {
            return LayerBuilder.Get(LayerEnum.Linear);
        }
        public static ILayer Relu()
        {
            return LayerBuilder.Get(LayerEnum.Relu);
        }

        public static ILayer SoftMax()
        {
            return LayerBuilder.Get(LayerEnum.SoftMax);
        }

        public static ILayer Sigmoid()
        {
            return LayerBuilder.Get(LayerEnum.Sigmoid);
        }

        public static ILayer TanH()
        {
            return LayerBuilder.Get(LayerEnum.TanH);
        }
    }
}
