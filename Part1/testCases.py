import unittest
import numpy as np

class TestAutogradSystem(unittest.TestCase):

    def test_addition_forward(self):
        x = Node(3)
        y = Node(4)
        z = x + y
        self.assertEqual(z.value, 7, "Addition forward pass failed")

    def test_addition_backward(self):
        x = Node(3)
        y = Node(4)
        z = x + y
        z.backward()
        self.assertEqual(x.grad, 1, "Addition backward pass for x failed")
        self.assertEqual(y.grad, 1, "Addition backward pass for y failed")

    def test_multiplication_forward(self):
        x = Node(3)
        y = Node(4)
        z = x * y
        self.assertEqual(z.value, 12, "Multiplication forward pass failed")

    def test_multiplication_backward(self):
        x = Node(3)
        y = Node(4)
        z = x * y
        z.backward()
        self.assertEqual(x.grad, 4, "Multiplication backward pass for x failed")
        self.assertEqual(y.grad, 3, "Multiplication backward pass for y failed")

    def test_graph_structure(self):
        # Test if the graph structure maintains correct parent/child relationships
        x = Node(3)
        y = Node(4)
        z = x * y + x
        self.assertIn(x, z.parents, "Graph structure does not contain x as a parent")
        self.assertIn(y, z.parents, "Graph structure does not contain y as a parent")

if __name__ == '__main__':
    unittest.main()
