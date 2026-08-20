# Team Peri Peri Fries - Team Peri Peri Fries v2 

This repository contains the Team Peri Peri Fries v2 forensic examination protocol engine, built by **Team Peri Peri Fries**.

## Training on a Separate GPU Laptop

To train the models (Stage A, B, and C) on a separate laptop equipped with a GPU (e.g., NVIDIA RTX 5060 or better), follow these instructions:

### 1. Transfer the Files
A training bundle has been pre-packaged for you. Transfer the `training_bundle.zip` file from the root of this repository to your training laptop. 

Extract the contents of the zip file into a new directory on the training laptop.

### 2. Setup the Environment
The training laptop requires CUDA 12.8 wheels for the latest NVIDIA GPUs (like the 50-series). 

Open a terminal on the training laptop and run the following commands:

```bash
# 1. Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install PyTorch with CUDA 12.8 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install the remaining requirements for the GPU track
pip install -r requirements-gpu.txt
```

### 3. Run the Training Scripts
Once the environment is configured, execute the training scripts in order. The models will be saved in the `artifacts/` directory upon completion.

```bash
# Train Stage A: Videoprint
python -m train.stage_a_videoprint

# Train Stage B: Decoder
python -m train.stage_b_decoder

# Train Stage C: Temporal
python -m train.stage_c_temporal
```

### 4. Handoff
After training is complete, transfer the generated model weights (`.pt` files) and their checksums (`.sha256` files) from the training laptop's `artifacts/` directory back to the `artifacts/` directory of this main repository.
