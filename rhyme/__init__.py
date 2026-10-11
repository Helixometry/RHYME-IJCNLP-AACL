"""RHYME: Generalizing Speech Deepfake Detection using Non-Euclidean Geometry"""

__version__ = "1.0.0"
__author__ = "Farhan Sheth, Girish, Mohd Mujtaba Akhtar, Muskaan Singh"
__email__ = "girish281003@gmail.com"

from .model import RHYME
from .geometry import HyperbolicProjection, SphericalProjection
from .encoders import MultiEncoderFusion

__all__ = [
    "RHYME",
    "HyperbolicProjection",
    "SphericalProjection",
    "MultiEncoderFusion",
]
