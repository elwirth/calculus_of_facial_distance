# Calculus of Facial Distance

This repository has been developped to explore facial distances and vertex distances over polytopes. 

---

## Installation & running a first example

1. Clone the repository.

2. Create and activate a Python environment:
    ```bash
    python -m venv venv_calc_of_fd
    source venv_calc_of_fd/bin/activate      # Linux/macOS
    # OR
    venv_calc_of_fd\Scripts\activate         # Windows
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the following code:
    ```bash
    python boolean_hypercube.py
    ```
    This creates results in `results`.

## Repository structure

```
.
├── boolean_hypercube.py        # Main example script
├── src/
│   ├── plotting_utils.py       # Plotting helper functions
│   └── polytope.py             # Polytope computations
├── results/                    # Saved plots and data
├── requirements.txt            # Python dependencies
└── README.md
```