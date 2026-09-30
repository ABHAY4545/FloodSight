import torch

# ── Dataset Constants ──────────────────────────────────────────────
NUM_CLASSES = 10

CLASS_NAMES = [
    "Background",
    "Building Flooded",
    "Building Non-Flooded",
    "Road Flooded",
    "Road Non-Flooded",
    "Water",
    "Tree",
    "Vehicle",
    "Pool",
    "Grass",
]

ID2COLOR = {
    0: (0, 0, 0),        # Background - Black
    1: (255, 0, 0),      # Building Flooded - Red
    2: (180, 120, 120),  # Building Non-Flooded - Dusty Rose
    3: (160, 150, 20),   # Road Flooded - Olive
    4: (140, 140, 140),  # Road Non-Flooded - Gray
    5: (61, 230, 250),   # Water - Cyan
    6: (0, 82, 255),     # Tree - Blue
    7: (255, 0, 245),    # Vehicle - Magenta
    8: (255, 235, 0),    # Pool - Yellow
    9: (4, 250, 7),      # Grass - Green
}

# ── Loss Hyperparameters ──────────────────────────────────────────

CLASS_WEIGHTS = torch.tensor([0.33, 1.00, 0.71, 0.76, 0.55, 0.38, 0.30, 2.98, 2.81, 0.17])
GAMMA = 2

# ── Normalization Stats ───────────────────────────────────────────

MEAN = (0.4102, 0.4471, 0.3415)
STD  = (0.2085, 0.1936, 0.2092)
