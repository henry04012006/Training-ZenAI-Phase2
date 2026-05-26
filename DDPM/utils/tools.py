import os
import torch
import torchvision
import matplotlib.pyplot as plt

def setup_directories() -> None:
    """Creates the necessary project directories (checkpoints/ and outputs/).
    
    Prevents runtime FileNotFoundError when saving intermediate output grids,
    loss curves, or training model weights.
    """
    directories = ["checkpoints", "outputs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def show_images(images: torch.Tensor, save_path: str = None, num_rows: int = 10) -> None:
    """Ingests a batch of generated images, un-normalizes, grids, saves (optional), and displays them.
    
    Args:
        images: Image tensor of shape [B, C, H, W] in the range [-1, 1].
        save_path: Optional destination file path to save the output grid image.
        num_rows: Target number of rows for the saved and displayed sheet.
    """
    # 1. Un-normalize from [-1, 1] to [0, 1]
    # Formula: (x + 1) / 2
    unnormalized_images = (images + 1.0) / 2.0
    
    # 2. Handle clipping with clamp(0, 1) to ensure valid pixel values
    clipped_images = torch.clamp(unnormalized_images, 0.0, 1.0)
    
    # 3. Calculate images per row (nrow) in make_grid to achieve exactly `num_rows` rows.
    # torchvision.utils.make_grid uses `nrow` as the number of columns (images per row).
    # Therefore, columns = total_images // num_rows.
    batch_size = clipped_images.size(0)
    if batch_size >= num_rows:
        nrow = batch_size // num_rows
    else:
        nrow = 1  # Fallback to single column if batch size is smaller than target rows
        
    # Create the visual grid sheet
    grid = torchvision.utils.make_grid(clipped_images, nrow=nrow, padding=2, pad_value=0.0)
    
    # 4. Save the grid to disk if a path is specified
    if save_path:
        torchvision.utils.save_image(grid, save_path)
        
    # 5. Display the grid on screen using matplotlib
    plt.figure(figsize=(10, 10))
    # permute channels from [C, H, W] to [H, W, C] for matplotlib compatibility
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    plt.imshow(grid_np)
    plt.axis("off")
    plt.show()

def plot_losses(losses: list[float], save_path: str = None) -> None:
    """Plots a line graph representing the convergence of the loss function.
    
    Args:
        losses: List of training loss values.
        save_path: Optional destination file path to save the generated plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training MSE Loss", color="royalblue", lw=2)
    plt.title("DDPM Training Loss Convergence", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epochs / Iterations", fontsize=12)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Save the plot to disk if a path is specified
    if save_path:
        plt.savefig(save_path, dpi=300)
        
    # Display the plot on screen
    plt.show()
