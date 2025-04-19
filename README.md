# Coordinate Descent

This repository provides PyTorch-based implementations of two coordinate descent optimization algorithms:
- Randomized Coordinate Descent (RCD)
- Steepest Coordinate Descent (SCD)

These optimizers can be used as drop-in replacements for standard PyTorch optimizers.

## Installation

1. Clone the repository:
   ```bash
   git clone <this-repo-url>
   cd coordinate-descent
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can use the optimizers just like any PyTorch optimizer:

```python
from src.rcd import RandomizedCoordinateDescent
from src.scd import SteepestCoordinateDescent

# Example: using RCD
optimizer = RandomizedCoordinateDescent(model.parameters(), lr=1e-3)

# Example: using SCD
optimizer = SteepestCoordinateDescent(model.parameters(), lr=1e-3)

for data, target in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

## Testing

Run the test suite with:
```bash
PYTHONPATH=./ pytest
```
This will run extensive tests on both RCD and SCD optimizers. To run specific tests, you can use:
```bash
PYTHONPATH=./ pytest tests/test_rcd.py::<test_name>
PYTHONPATH=./ pytest tests/test_scd.py::<test_name>
```

Configuration files are stored in the `tests/config` directory.

## References
1. [PyTorch Documentation](https://pytorch.org/docs/stable/optim.html)
2. [Lecture Notes](https://n.ethz.ch/~jiaxie/graduate_projs/notes_eth.pdf), Optimization in Data Science, FS23, ETH Zurich