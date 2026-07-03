import unittest
import omage as og


class TestLayers(unittest.TestCase):
    
    def test_dense(self):
        layer = og.dense(128)
        self.assertEqual(layer.units, 128)
        self.assertEqual(layer.activation, "relu")
    
    def test_dense_custom_activation(self):
        layer = og.dense(64, activation="sigmoid")
        self.assertEqual(layer.activation, "sigmoid")
    
    def test_dropout(self):
        layer = og.dropout(0.3)
        self.assertEqual(layer.rate, 0.3)
    
    def test_conv2d(self):
        layer = og.conv2d(32, kernel_size=3)
        self.assertEqual(layer.filters, 32)
        self.assertEqual(layer.kernel_size, 3)


if __name__ == "__main__":
    unittest.main()