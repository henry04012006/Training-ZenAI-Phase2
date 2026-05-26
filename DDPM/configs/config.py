import torch

class DiffusionConfig:
    """Configuration class for DDPM (Denoising Diffusion Probabilistic Models) training.
    
    This class encapsulates dataset dimensions, optimization parameters, diffusion process
    coefficients, and compute device configuration, serving as the central source of truth
    for the model training and sampling pipeline.
    """
    
    # Dataset & Structural Dimensions
    IMG_SIZE: int = 32          # Spatial resolution of MNIST images after resizing (height and width)
    CHANNELS: int = 1           # Number of input channels (1 for grayscale MNIST)
    NUM_CLASSES: int = 10       # Digit classes (0-9) used for Classifier-Free Guidance (CFG)
    
    # Optimization & Training Controls
    BATCH_SIZE: int = 128       # Training mini-batch size
    EPOCHS: int = 25            # Number of full training epochs
    LR: float = 2e-4            # Learning rate (Adam optimizer)
    
    # Diffusion Process Coefficients
    TIMESTEPS: int = 1000       # Total number of discrete diffusion steps (T)
    BETA_START: float = 1e-4    # Initial noise level variance (beta_1)
    BETA_END: float = 0.02      # Final noise level variance (beta_T)
    
    # Hardware Acceleration & Platform Detection
    DEVICE: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
