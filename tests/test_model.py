"""Unit tests for RHYME model."""

import unittest
import torch
from rhyme.model import RHYME


class TestRHYME(unittest.TestCase):
    """Test cases for RHYME model."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = RHYME(
            input_dim=1024,
            hidden_dim=512,
            dropout=0.3,
        )
        self.batch_size = 4
        self.input_dim = 1024

    def test_model_initialization(self):
        """Test model can be initialized."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.input_dim, 1024)

    def test_forward_pass(self):
        """Test forward pass."""
        x = torch.randn(self.batch_size, self.input_dim)
        output = self.model(x)

        self.assertIn('deepfake', output)
        self.assertIn('generator', output)
        self.assertIn('features', output)

        self.assertEqual(output['deepfake'].shape, (self.batch_size, 2))
        self.assertEqual(output['generator'].shape, (self.batch_size, 5))

    def test_model_save_load(self):
        """Test model save and load."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')

            # Save model
            self.model.save(checkpoint_path)
            self.assertTrue(os.path.exists(checkpoint_path))

            # Load model
            loaded_model = RHYME.from_pretrained(checkpoint_path)
            self.assertIsNotNone(loaded_model)


if __name__ == '__main__':
    unittest.main()
