"""
Advanced Neural Network - Python Package
"""

__version__ = "1.0.0"
__author__ = "Neural Network Team"
__description__ = "Advanced Neural Network with Dynamic Text Generation"

from .neural_network import NeuralNetwork
from .tokenizer import Tokenizer
from .generator import TextGenerator
from .config import GenerationConfig

__all__ = [
    "NeuralNetwork",
    "Tokenizer", 
    "TextGenerator",
    "GenerationConfig",
]

