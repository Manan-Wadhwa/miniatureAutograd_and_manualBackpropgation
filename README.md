# Autograd System Notebook

## Project Overview
This notebook provides a Python implementation of an autograd system designed for differentiating functions and calculating gradients automatically. The system uses a series of nodes to compute forward and backward passes in a computational graph. The autograd mechanics enable dynamic computation of gradients, making it applicable to machine learning tasks, especially for neural network training.

## Features
- **Node-Based Computational Graph**: Each operation is represented by nodes that store values and gradients.
- **Forward and Backward Passes**: Automatic differentiation is achieved through forward computation and backpropagation.
- **Gradient Calculation**: Nodes record gradients for their inputs, enabling the calculation of derivatives with respect to various variables.

## File Contents
1. **Autograd Class** - Defines core mechanics for creating nodes and handling gradient propagation.
2. **Node Functions** - Implements tensor operations (addition, multiplication) with gradient support.
3. **Graph Operations** - Includes utility methods for managing nodes and connections in the computational graph.

## Prerequisites
- Python 3.x
- Numpy
- Pytorch

## How to Use
1. Open the notebook file.
2. Run all cells sequentially to execute the autograd implementation.
3. Modify the input values in the sample cells to experiment with different forward and backward passes.

## Test Cases
For testing this notebook’s functionality, the following tests are essential:

1. **Gradient Calculation**
   - Check if the gradient calculation is correct for basic arithmetic operations, like addition and multiplication.
   - Test scenarios:
     - Forward pass for simple addition (`x + y`) should yield correct output and gradients.
     - Forward pass for multiplication (`x * y`) should yield correct output and gradients.

2. **Backward Pass**
   - Validate that the backward pass computes the expected gradients for each node.
   - Test scenarios:
     - Apply backpropagation on composite functions to ensure chain rule compliance (e.g., `f(x, y) = x * y + z`).
  
3. **Graph Structure Validation**
   - Confirm that nodes correctly link to their parents and children in the graph.
   - Test scenarios:
     - After adding nodes, verify the computational graph structure.

4. **Edge Cases**
   - Test handling of edge cases, such as:
     - Division by zero or undefined operations.
     - Non-differentiable points (e.g., zero for multiplication).

These tests can be implemented using Python’s `unittest` or `pytest` to ensure accurate function and gradient calculations across various cases.
