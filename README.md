# Light Weight Machine Learning Library

This projects can be used as a tiny library to create and use neural networks.

At the moment, only deep neural networks can be implemented, but hopefully with time we'll go further and develop more types of of networks !

## Classes

### Activation

Static class that provides the `Layer` class with activation functions and their respective derivatives.

The library currently supports :

- TanH
- Sigmoid
- ReLU


### Layer

A `Layer` stores inputs, weights, biases and outputs.

It performs the mathematical operations to propagate data forward or backward, and adjust its biases and weights through gradient descent.

When propagating data, either forward or backward, its returned output is mapped through activation function (going forward) or activation function's derivative (backward).

**Note that the stored output doesn't get modified by the mapping operation**

When initialized, it must be provided with three parameters :

- `int` Number of input nodes
- `int` Number of output nodes
- `ActivationEnum` The enum of the function used as activation

Example :

```C#
Layer layer = new Layer(2, 3, ActivationEnum.Sigmoid);
```

### NeuralNetwork

A `NeuralNetwork` stores `Layers`.

When training, it feeds data forward through it's layer, then computes the error (we use mean squared error), then backpropagate the errors.

It must be provided with layers on initialization (no need to pack them into an array).

**Note that the number of input neuron in a given layer must be the same as the number of output nodes in the previous layer**

**Inconsistency in layer shapes is not handled at initialization an will result in errors when performing matrix operations**

Example:

```C#
NeuralNetwork nn = new NeuralNetwork(
	new Layer(2, 3, ActivationEnum.Sigmoid)
	new Layer(3, 1, ActivationEnum.Sigmoid));
```

It will train on the given `DataPoint[]` for as many `epochs` and with the `learningRate` provided as argument to its `Train` function.

```C#
nn.Train(dataSet, 10000, 0.1);
```

You can then use it to make predictions, returned as `double[]`.

```C#
nn.Predict(dataPoint, 10000, 0.1);
```

## Utils

### ActivationEnum

This `enum` stores the currently available activation functions of the library.

It should be used to feed the layers third parameters on initialization.

### DataPoint

Simple `struct` to hold `X` input and `Y` output for a single point of data.

It should be used to build or map the dataset.

Example:

```c#
DataPoint[] xorDataSet = new DataPoint[]
{
	new DataPoint(new double[]{ 0.0, 0.0 }, new double[]{ 0.0 }),
	new DataPoint(new double[]{ 1.0, 1.0 }, new double[]{ 0.0 }),
	new DataPoint(new double[]{ 1.0, 0.0 }, new double[]{ 1.0 }),
	new DataPoint(new double[]{ 0.0, 1.0 }, new double[]{ 1.0 })
}
```

## Example with XOR problem

```c#
using MachineLearning;

NeuralNetwork nn = new NeuralNetwork(
    new Layer(2, 3, ActivationEnum.Sigmoid),
    new Layer(3, 1, ActivationEnum.Sigmoid));

DataPoint[] xorDataSet = new DataPoint[]
{
    new DataPoint(new double[] { 0.0, 0.0 }, new double[] { 0.0}),
    new DataPoint(new double[] { 1.0, 0.0 }, new double[] { 1.0}),
    new DataPoint(new double[] { 0.0, 1.0 }, new double[] { 1.0}),
    new DataPoint(new double[] { 1.0, 1.0 }, new double[] { 0.0})
};

nn.Train(xorDataSet, 1000, 0.1);

Console.WriteLine(String.Format("{0:0.00}", nn.Predict(new double[] { 0.0, 0.0 })[0]));
Console.WriteLine(String.Format("{0:0.00}", nn.Predict(new double[] { 1.0, 1.0 })[0]));
Console.WriteLine(String.Format("{0:0.00}", nn.Predict(new double[] { 1.0, 0.0 })[0]));
Console.WriteLine(String.Format("{0:0.00}", nn.Predict(new double[] { 0.0, 1.0 })[0]));
```
