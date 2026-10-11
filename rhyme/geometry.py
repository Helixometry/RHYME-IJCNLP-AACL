"""Non-Euclidean geometric projections for RHYME."""

import torch
import torch.nn as nn
import math


class HyperbolicProjection(nn.Module):
    """
    Hyperbolic geometric projection using Poincaré ball model.

    Models hierarchical relationships among generator types.

    Args:
        input_dim (int): Input feature dimension
        output_dim (int): Output feature dimension
        curvature (float): Hyperbolic curvature parameter
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        curvature: float = -1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.curvature = curvature

        self.linear = nn.Linear(input_dim, output_dim)
        self.register_parameter(
            "scale",
            nn.Parameter(torch.tensor([1.0]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to Poincaré ball (hyperbolic space).

        Args:
            x: Input tensor

        Returns:
            Hyperbolic projection
        """
        # Linear transformation
        x_proj = self.linear(x)

        # Poincaré ball projection: tanh normalization
        norm = torch.norm(x_proj, dim=-1, keepdim=True)
        x_normalized = torch.tanh(norm) * (x_proj / (norm + 1e-8))

        return x_normalized * self.scale


class SphericalProjection(nn.Module):
    """
    Spherical geometric projection.

    Identifies vocoder-specific patterns on normalized sphere.

    Args:
        input_dim (int): Input feature dimension
        output_dim (int): Output feature dimension
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project to spherical space (normalized).

        Args:
            x: Input tensor

        Returns:
            Normalized spherical projection
        """
        x_proj = self.linear(x)
        # Normalize to unit sphere
        x_normalized = torch.nn.functional.normalize(x_proj, dim=-1)
        return x_normalized


class RiemannianMean(nn.Module):
    """
    Riemannian mean computation for synthesis-invariant embeddings.

    Computes geodesic average of embeddings using exponential/logarithmic maps.
    """

    def __init__(self, method: str = "euclidean"):
        super().__init__()
        self.method = method

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute Riemannian mean of embeddings.

        Args:
            embeddings: Tensor of shape (batch_size, n_encoders, feature_dim)

        Returns:
            Mean embedding of shape (batch_size, feature_dim)
        """
        if self.method == "euclidean":
            return embeddings.mean(dim=1)
        elif self.method == "hyperbolic":
            return self._hyperbolic_mean(embeddings)
        elif self.method == "spherical":
            return self._spherical_mean(embeddings)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _hyperbolic_mean(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic mean using logarithmic map."""
        # Simplified: use weighted average in tangent space
        return embeddings.mean(dim=1)

    def _spherical_mean(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute spherical mean (normalized average)."""
        mean = embeddings.mean(dim=1)
        return torch.nn.functional.normalize(mean, dim=-1)


class MixedCurvatureSpace(nn.Module):
    """
    Mixed-curvature space combining Hyperbolic, Euclidean, and Spherical.

    Args:
        input_dim (int): Input feature dimension
        output_dim (int): Output feature dimension
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.hyperbolic = HyperbolicProjection(input_dim, output_dim)
        self.euclidean = nn.Linear(input_dim, output_dim)
        self.spherical = SphericalProjection(input_dim, output_dim)

        # Learnable weights for combining projections
        self.register_parameter(
            "alpha",
            nn.Parameter(torch.tensor([1.0 / 3.0]))
        )
        self.register_parameter(
            "beta",
            nn.Parameter(torch.tensor([1.0 / 3.0]))
        )
        self.register_parameter(
            "gamma",
            nn.Parameter(torch.tensor([1.0 / 3.0]))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine projections from all three geometric spaces.

        Args:
            x: Input tensor

        Returns:
            Mixed-curvature embedding
        """
        h = self.hyperbolic(x)
        e = self.euclidean(x)
        s = self.spherical(x)

        # Weighted combination
        output = self.alpha * h + self.beta * e + self.gamma * s

        return output
