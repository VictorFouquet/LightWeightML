
namespace MachineLearning
{
    public struct DataPoint
    {
        public double[] X;
        public double[] Y;
        public DataPoint(double[] X, double[] Y)
        {
            this.X = X;
            this.Y = Y;
        }
    }
}
