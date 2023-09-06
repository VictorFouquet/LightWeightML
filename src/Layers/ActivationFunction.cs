namespace MachineLearning.Layers
{
    internal class ActivationFunction
    {
        public static double Linear(double x)
        {
            return x;
        }

        public static double LinearDerivative(double x)
        {
            return 1;
        }

        public static double Relu(double x)
        {
            return Math.Max(0, x);
        }

        public static double ReluDerivative(double x)
        {
            return x > 0 ? 1 : 0;
        }

        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public static double SigmoidDerivative(double x)
        {
            var sigX = Sigmoid(x);
            return sigX * (1 - sigX);
        }

        public static double TanH(double x)
        {
            return Math.Tanh(x);
        }

        public static double TanHDerivative(double x)
        {
            return 1 - Math.Pow(Math.Tanh(x), 2);
        }
    }
}
