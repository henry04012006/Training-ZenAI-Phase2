# Project Context & Blueprint: DDPM from Scratch

## 1. Absolute Assignment Requirements & Grading Criteria
To ensure maximum score and strict compliance with the assignment rules, the AI Agent must follow these constraints perfectly:
* **Core Goal:** Gain a profound, fundamental understanding of how Diffusion Models work by building everything from the ground up.
* **Framework Constraint:** Use **PyTorch ONLY**. The use of the `diffusers` library or any pre-built diffusion pipelines/wrappers is **strictly prohibited**.
* **Dataset:** MNIST (Grayscale handwritten digits, 1x28x28).
* **Core Components Required:**
    1. **Noise Scheduler:** Implement both Linear and Cosine variance schedules to compute $\beta_t$, $\alpha_t$, and $\bar{\alpha}_t$.
    2. **Forward Process:** Implement the explicit algebraic function `q_sample(x0, t, noise)`.
    3. **Model Architecture:** Build a minimal U-Net utilizing custom ResNet blocks paired with Sinusoidal Time Embeddings.
    4. **Training Loop:** A native PyTorch Trainer executing timestep sampling, noise injection, noise prediction, and MSE optimization.
    5. **Sampling/Inference Loop:** Reverse Markov chain denoising loop starting from pure Gaussian noise $x T$ over $T$ steps.
* **Advanced Bonus Target:** Add class conditioning utilizing **Classifier-Free Guidance (CFG)** based on digit labels (0–9) and visualize class-specific conditional generations.
* **Mandatory Outputs:**
    - Clean, isolated, modular code blocks for every component.
    - Training performance demonstrated via clear loss logging and an explicit MSE loss curve chart.
    - Visual grid of random images synthesized purely from latent noise.

---

## 2. Global Execution Workflow
The implementation of this project follows a strict chronological modular sequence:
* **Phase 1: Project Configurations (`configs/config.py`)** $\leftarrow$ *CURRENT STEP*
* **Phase 2:** Hardware-agnostic Utilities & Visualizations (`utils/tools.py`)
* **Phase 3:** Structural Neural Network Foundations (`models/blocks.py` & `models/unet.py`)
* **Phase 4:** Core Diffusion Mathematical Framework (`models/diffusion.py`)
* **Phase 5:** Integrated Pipeline Vouching (Data, Training, Sampling in `main.ipynb`)

---

## 3. Phase 1: System Configurations & Hyperparameters

### 3.1 Technical Background & Mathematical Rationale
Following the foundational DDPM paper (*Ho et al., 2020*), diffusion models require a precise scheduling of variance boundaries to successfully convert clean data distributions into standard Gaussian noise. 
* **The Variance Boundaries:** $\beta_t$ dictates the scale of noise injected at each forward step. The paper establishes a linear schedule where $\beta$ increments from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$.
* **The Timestep Scale ($T$):** Set globally to $1000$ steps. This ensures that the discrete transition approximation closely maps to a continuous reverse thermodynamic process, preventing distribution shifts during backward sampling.
* **MNIST Resolution Realities:** MNIST contains low-frequency spatial profiles (1x28x28 grayscale). While $T=1000$ guarantees peak generation fidelity, config parameters should remain cleanly decoupled so that $T$ or network channels can be minimized seamlessly during local test runs.

### 3.2 Configuration Architecture Design (`configs/config.py`)
To isolate hyperparameter state tracking away from logical computation code, the first step of the workflow defines a static configuration class structure. It encapsulates three distinct logic pools:

#### A. Dataset & Structural Dimensions
* `IMG_SIZE = 32`: Height and width matching standard MNIST dimensions after resizing.
* `CHANNELS = 1`: Single channel for grayscale tracking.
* `NUM_CLASSES = 10`: Labels ranging from 0 to 9 for the advanced class-conditioning bonus.

#### B. Optimization & Training Controls
* `BATCH_SIZE = 128`: Balanced batch constraint for optimal gradient step stochasticity on local GPUs.
* `EPOCHS = 20`: Sufficient training epochs to ensure clear loss convergence and legible structural digit synthesis.
* `LR = 2e-4`: Standard stable learning rate recommended for architectural training variants of U-Net score-matchers.

#### C. Diffusion Process Coefficients
* `TIMESTEPS = 1000`: The complete forward discrete horizon ($T$).
* `BETA_START = 1e-4`: Bottom variance threshold ($\beta_1$).
* `BETA_END = 0.02`: Upper variance threshold ($\beta_T$).

#### D. System Environment
* Runtime platform compute target acceleration detection logic mapping to `cuda` (NVIDIA), `mps` (Apple Silicon), or falling back to `cpu`.

---
## 4. Phase 2: Hardware-agnostic Utilities & Visualizations (`utils/tools.py`)

This section defines the structural blueprint for helper utilities responsible for tracking training metrics, enforcing clean environmental setups, and handling the visualization grids required by the assignment evaluation criteria.

### 4.1 Directory Auto-Generation Logic
To preserve high portability and prevent runtime crashes (`FileNotFoundError`), the helper file must implement an initialization check to dynamically establish target directories on the operating system.
* **Target Paths:** 1. `checkpoints/`: Dedicated folder for storing network weight matrices (`.pth` or `.pt` files).
  2. `outputs/`: Consolidated location for generated qualitative image sheets and analytical evaluation curves.
* **Implementation Standard:** Use standard Python `os.makedirs(..., exist_ok=True)` inside a core environmental setup helper function.

### 4.2 Metrics Tracking and Loss Convergence Curves
The assignment explicitly dictates that training progression must be verified using metric tracking metrics or structured visual plots.
* **Mathematical Rationale:** Over training iterations, the mean squared error ($L_{\text{simple}}$) between $\epsilon_{\text{target}}$ and $\epsilon_\theta$ should smoothly decrease, plateuing as score-matching distribution alignment completes.
* **Agent Implementation Rule:** - Function Name: `plot_loss_curve(loss_history: list[float], save_path: str = "outputs/loss_curve.png") -> None`
  - Internal Details: Utilize `matplotlib.pyplot` to generate a line graph representing Average MSE Loss over Epochs/Steps. Ensure proper labeling (`X-axis: Epochs/Steps`, `Y-axis: MSE Loss`, and a descriptive plot title). The chart must be saved directly to the configured path before rendering inside the main pipeline.

### 4.3 Generated Quantization Visual Grids (Image Sampling Utilities)
To fulfill the generation requirements, the system requires a rigid visualization engine capable of re-scaling continuous raw model outputs into human-legible digital image grids.
* **The Scale Shift Realities:** The internal U-Net operates across normalized continuous tensor boundaries of `[-1, 1]`. For standard image rendering tools (`matplotlib` or `torchvision.utils.save_image`), this space must be strictly un-normalized back to the discrete pixel range `[0, 1]`.
  $$\hat{x} = \frac{x + 1}{2}$$
* **Advanced Conditional Grid Alignment (For CFG Bonus Visualization):**
  - Function Name: `save_and_display_grid(images: torch.Tensor, save_path: str, num_rows: int = 10) -> None`
  - Dynamic Layout Rule: When sample evaluation completes, images should be arranged logically. For conditional sampling verification, structure the grid layout specifically into 10 rows matching class indices `0` to `9`. Each row must explicitly cluster images synthesized under that identical label condition, providing perfect transparency for conditional generation assessment.
  - Processing Details: Ingest a continuous tensor of shape `[B, 1, 32, 32]`, perform the pixel conversion math (`clamp(0, 1)` to prevent arithmetic clipping artifacts), and call `torchvision.utils.make_grid` or equivalent native matrix transformations to combine elements into a unified grid canvas. Save the artifact inside `outputs/` and render the final grid preview cell seamlessly inside the runtime environment.

---

## 5. Phase 3: Core Neural Architecture & Mathematical Implementation (`models/`)

This section consolidates the core mathematical mechanics and network architectures of the Denoising Diffusion Probabilistic Model (DDPM) optimized for Classifier-Free Guidance (CFG). By resizing the raw MNIST data pipeline to a standardized $32 \times 32$ matrix footprint, the symmetrical neural backbone is mathematically structured to undergo clean resolution divisions across exactly four distinct spatial scales ($32 \rightarrow 16 \rightarrow 8 \rightarrow 4$).

### 5.1 Basic Structural Components, Time & Class Conditioning (`models/blocks.py`)
Because the score-matching network must dynamically modulate its filters based on the continuous variance level of the input data and the conditional class label at any arbitrary step $t$, specialized architectural blocks are engineered to support progressive dual conditioning.

1. **Sinusoidal Time Embedding**
   * **Theoretical Background:** Inherited from Transformer architectures, time step scalars $t \in [0, T-1]$ are mapped into continuous dense vector representations using varying frequency sinusoidal functions:
     $$\text{PE}_{(t, 2i)} = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}_{(t, 2i+1)} = \cos\left(\frac{t}{10000^{2i/d}}\right)$$
     where $d$ is the embedding dimension. This permits a single shared U-Net parameter pool to process arbitrary noise levels while adjusting its internal feature activation profiles.
   * **Implementation Blueprint:** Define a `SinusoidalPositionEmbeddings` module that maps input time tensors into a distinct `time_dim` projection space, followed by a two-layer Multi-Layer Perceptron (MLP) with `SiLU` activation to yield the final time conditioning vector.

2. **Class Conditioning & Classifier-Free Guidance (CFG) Embedding**
   * **Theoretical Background:** To perform class-conditioned generation with CFG, the model must accept both conditional class labels $c \in [0, 9]$ and a special unconditional/null label symbol $\emptyset$. During training, we drop out conditioning with a probability $p_{\text{uncond}} = 0.1 \sim 0.2$ to let the network learn both conditioned and unconditioned representations.
   * **Implementation Blueprint:** 
     * **Embedding Table:** Instantiate an `nn.Embedding(11, class_dim)` representing 11 classes: indices `0-9` for digits, and index `10` for the unconditional/null token $\emptyset$.
     * **Conditioning Injection:** Project the class embedding via a linear layer and combine it with the time embedding vector (e.g., via element-wise addition) to form a unified conditioning vector:
       $$\text{cond\_emb} = \text{time\_emb} + \text{class\_emb}$$
     * **Conditioning Dropout:** Implement training logic to dynamically mask labels to index `10` based on a predefined dropout probability.

3. **Residual Network (ResNet) Blocks with Conditioning Injection**
   * **Implementation Specifications:**
     * **Group Normalization (`GroupNorm`):** Deployed across all convolutional layers instead of `BatchNorm` to decouple batch size dependency. Rule: Set `num_groups = 32`. Ensure that channel dimensions (e.g. 64, 128, 256, 512) are strictly divisible by 32 to prevent runtime division crashes.
     * **Temporal/Class Injection Pipeline:** The unified conditioning vector `cond_emb` is projected through a linear layer and added element-wise (or applied via scale-shift) directly to the feature maps after the first convolution block:
       $$h_{\text{cond}} = h + \text{Linear}(\text{cond\_emb})[:, :, \text{None}, \text{None}]$$
     * **Non-linear Activation:** Utilize `SiLU` (Swish) activation for smooth gradients.
     * **Shortcut Connection:** If the input channels do not match the output channels, apply a $1 \times 1$ convolution to match dimensions before the residual addition.

### 5.2 Deep Symmetrical Architecture Assembly (`models/unet.py`)
The functional role of the localized U-Net backbone is to approximate the parameterized noise estimator function $\epsilon_\theta(x_t, c, t)$, learning a mapping from a noisy sample tensor $x_t$ back to its clean target perturbation variance $\epsilon$.

* **Down-path (Contracting Stream):** Receives the initial $32 \times 32$ single-channel tensors. It utilizes exactly 3 spatial reduction operations (parameterized via `Conv2d` with a stride of 2 or `MaxPool2d`), interleaved with continuous `ResNetBlock` primitives.
  * *Scale 1:* $32 \times 32$ spatial dimensions.
  * *Scale 2:* $16 \times 16$ spatial dimensions. **Self-Attention Requirement:** To capture non-local structural configurations and macro-digit semantics, a Multi-Head Self-Attention module is strictly integrated at this resolution boundary.
  * *Scale 3:* $8 \times 8$ spatial dimensions.
* **Bottleneck (Latent Matrix Base):** Reaches the deep structural floor at a localized resolution of $4 \times 4$. The architecture at this stage encapsulates two sequential `ResNetBlocks` separated by a central Multi-Head Self-Attention core layer.
* **Up-path (Expanding Stream) & Dimension Matching Rules:**
  * Progressively scales latent representations back up through structural dimensions ($4 \times 4 \rightarrow 8 \times 8 \rightarrow 16 \times 16 \rightarrow 32 \times 32$) using spatial `nn.Upsample` transformations.
  * **Skip Connections Concat Rule:** At each expansion step, incoming feature arrays are structurally concatenated with their exact spatial counterpart from the contracting down-path stream via `torch.cat([up_features, down_features], dim=1)`.
  * **Channel Tracking:** Because concatenation doubles the incoming channel size, the subsequent `ResNetBlock` must be initialized with `in_channels = C_up_in + C_down_out`. For instance, if the upsampled feature maps have 256 channels and the skip connection provides 256 channels, the ResNet block must accept 512 input channels and output 256 channels.

### 5.3 Mathematical Diffusion Engines (`models/diffusion.py`)
This script implements the core probabilistic frameworks governing both the forward data corruption and reverse iterative sampling schedules.

1. **Variance and Noise Schedulers**
   * **Linear Schedule (Baseline):** Linearly increments the variance coefficients from a safe initial boundary $\beta_1 = 10^{-4}$ up to an upper noise boundary $\beta_T = 0.02$ across the discrete time horizon $T=1000$.
   * **Cosine Schedule (Advanced Variant):** Implements a smooth variance trajectory utilizing a cosine formulation to limit aggressive data corruption at the terminal boundaries, calculated via $\bar{\alpha}_t = \frac{f(t)}{f(0)}$ where $f(t) = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$ and $s=0.008$.
   * **Numerical Stability Precaution:** During division steps, mẫu số (denominators) containing $1 - \bar{\alpha}_t$ are clamped to prevent arithmetic division-by-zero crashes:
     $$\text{denom\_safe} = \sqrt{\max(1 - \bar{\alpha}_t, 10^{-8})}$$

2. **Forward Process Execution (`q_sample`)**
   * **Mathematical Formulation:** Employing the Gaussian distribution reparameterization trick, any corrupted state $x_t$ can be sampled in closed-form at an arbitrary point $t$ directly from the uncorrupted base image data $x_0$:
     $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$

3. **Training Objective Formulation (`get_loss`)**
   * **Classifier-Free Guidance Training:** During training, classes are randomly dropped out with $p_{\text{uncond}} = 0.1$. The loss simple objective minimizes:
     $$L_{\text{simple}} = \mathbb{E}_{t, x_0, c, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, c_{\text{masked}}, t) \|^2 \right]$$
     where $c_{\text{masked}} = 10$ with probability $p_{\text{uncond}}$, and $c$ otherwise.
   * **Procedural Flow:** Extract a batch of clean source images $x_0$ and class labels $c$, sample a random sequence of integers $t$, generate a standard normal noise tensor $\epsilon$, compute the analytical noisy representation $x_t$ via `q_sample`, pass the resulting tensor, time coordinates, and masked class labels into the structural U-Net to yield $\epsilon_\theta$, and return the computed loss using `F.mse_loss(epsilon, epsilon_theta)`.

4. **Reverse Process Generation with Classifier-Free Guidance (`sample`)**
   * **Classifier-Free Guidance Formulation:** The guided noise estimation is calculated by extrapolating the conditioned noise estimation away from the unconditioned noise estimation:
     $$\tilde{\epsilon}_\theta(x_t, c, t) = (1 + w) \epsilon_\theta(x_t, c, t) - w \epsilon_\theta(x_t, \emptyset, t)$$
     where $w \ge 0$ is the guidance scale parameter (typically $2.0 \sim 3.0$ for MNIST)
   * **Markov Chain Step:** Beginning from pure isotropic Gaussian latent arrays $x_T \sim \mathcal{N}(0, I)$, the model moves backward step-by-step toward the uncorrupted data manifold using the parameterized reverse transitions:
     $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\text{denom\_safe}} \tilde{\epsilon}_\theta(x_t, c, t) \right) + \sigma_t z$$
   * **Parameter Details:**
     * $z \sim \mathcal{N}(0, I)$ represents standard random noise. When tracking down to the terminal step ($t=0$), $z$ is explicitly zeroed out ($z=0$).
     * The stochastic variance coefficient $\sigma_t$ is assigned natively as $\sqrt{\beta_t}$

--
## 6. Phase 4: System Coordination, Training Lifecycle & Sampling Execution (`main.ipynb`)

This section codifies the operational execution pipeline within the main notebook. Serving as the primary entry point, `main.ipynb` orchestrates the system by importing modular architectures, managing the data loading mechanics, driving the iterative optimization loop, and executing the conditioned generation phases.

### 6.1 Unified Imports & Workspace Instantiation
The initialization interface acts as a central coordinator, systematically fetching dependencies from isolated source code files without pipeline wrappers:
* **System Packages:** Core PyTorch libraries (`torch`, `torch.nn`, `torch.optim`) and `torchvision`.
* **Project Specifications:** Global static hyperparameter structures extracted directly from `configs.config`.
* **Network Graph Components:** The unified score estimator `UNet` from `models.unet` and the mathematical diffusion engine `DDPM` from `models.diffusion`.
* **Analytical Helpers:** Environment setup routines, tracking utilities, and grid generators from `utils.tools`.

### 6.2 Data Ingestion & Space Normalization Pipelines
The processing pipeline converts raw digit matrices into arrays optimized for continuous Gaussian noise addition.
* **Spatial Alignment:** Applies a structural resize operation mapping the raw $28 \times 28$ MNIST canvas size directly to a standardized $32 \times 32$ matrix domain. This modification ensures perfect downscaling/upscaling factorizations across the four discrete architectural scales ($32 \rightarrow 16 \rightarrow 8 \rightarrow 4$).
* **Range Normalization:** Uses a continuous tensor transformation mapping via `transforms.Normalize((0.5,), (0.5,))` to project raw data from pixel boundaries $[0, 1]$ directly into the zero-centered symmetrical range $[-1, 1]$. This normalization stabilizes variance matching when tracking score predictions against standard normal distributions $\mathcal{N}(0, I)$.
* **Streaming Mechanics:** Encapsulates the operations inside a standard `DataLoader`, managing uniform shuffling, workers, and packaging structures bounded by the target `BATCH_SIZE`.

### 6.3 Optimization Engine & Training Dynamics
The training loop converts conceptual diffusion mechanics (Algorithm 1) into active matrix optimizations to fit the network parameters $\theta$.

* **Graph Setup:** Instantiates the structural `UNet` core and the mathematical `Diffusion` scheduler independently, transfers both to the accelerated target computing `device`, and binds the model parameters to a standard `torch.optim.Adam` optimizer governed by the target learning rate (`LR`).
* **The Iterative Optimization Loop:**
  ```python
  # Ensure models are on the proper execution device
  model = UNet(...).to(device)
  diffusion = Diffusion(...).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)

  for epoch in range(EPOCHS):
      for batch_images, batch_labels in dataloader:
          optimizer.zero_grad()
          
          # Move training batch tensors to the designated accelerator device
          batch_images = batch_images.to(device)
          batch_labels = batch_labels.to(device)
          
          # 1. Sample dynamic temporal coordinates uniformly across the discrete horizon
          t = torch.randint(0, TIMESTEPS, (batch_images.shape[0],), device=device)
          
          # 2. Compute error metrics under conditional/unconditional drop constraints (CFG)
          # Correct parameters: (model, images, labels, timesteps)
          loss = diffusion.get_loss(model, batch_images, batch_labels, t)
          
          # 3. Propagate error matrices and adjust node weights
          loss.backward()
          optimizer.step()
  ```

--
## Current Phase Checklist
- [x] Create `configs/` folder and write a clean, hyperparameter-isolated `config.py`.
- [x] Ensure config attributes are typed correctly and imported cleanly inside other modules.
- [x] Create `utils/` folder and write `tools.py` containing directory setup, loss curve plotting, and image grid saving.
- [x] Verify directory setup (`checkpoints/`, `outputs/`) works, loss curve plots correctly, and grids are un-normalized and clamped.
- [x] Create `models/blocks.py` with SinusoidalPositionEmbeddings, ResNetBlock with conditioning, and SelfAttention modules.
- [x] Create `models/unet.py` with symmetrical U-Net architecture, contracting/expanding streams, and dimension matching skip connections.
- [x] Create `models/diffusion.py` with linear/cosine noise schedulers, forward process, reverse process with CFG, and training loss functions.
- [x] Verify compilation and execution of U-Net and Diffusion models on test tensors.
- [x] Create `main.ipynb` containing the unified data pipeline, model instantiations, training loops, loss plotting, and conditional CFG sampling.
- [x] Verify that the notebook syntax is valid and ready to run on target accelerator devices.
- [x] Optimize U-Net capacity by setting `base_channels = 128` for high-fidelity MNIST generation.
- [x] Integrate `schedule_type = "cosine"` inside `main.ipynb` to smoothly distribute noise coefficients.
- [x] Structure `main.ipynb` with clear Markdown headings and embed epoch-wise intermediate sampling visual inspections.
