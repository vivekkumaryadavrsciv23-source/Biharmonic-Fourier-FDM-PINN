# Efficient PINN Architectures for Kirchhoff Plate Bending

> **Associated Paper:**  
> *"Comparative Assessment of Physics-Informed Neural Network Architectures with Hard Boundary Constraints for Kirchhoff Plate Bending: Vanilla, FDM-Hybrid, Fourier-Enhanced, and Fourier-FDM Formulations"*  
> Vivek Kumar Yadav, Kshitij Kumar Yadav, Rajesh Kumar  
> Department of Civil Engineering, Indian Institute of Technology (BHU), Varanasi

---

## 📌 Overview

This repository contains the complete implementation of a **40-case systematic benchmark** comparing four Physics-Informed Neural Network (PINN) architectures for solving the **Kirchhoff–Love thin plate bending problem** (biharmonic PDE).

**Key Results:**
- FDM-PINN is **13–56× faster** than Vanilla PINN with **37× less GPU memory**
- Fourier-FDM PINN achieves the **lowest bending-moment errors** across all cases
- Hard boundary constraints guarantee **exact BC satisfaction** — no penalty tuning needed

---

## 🏗️ Architectures Compared

| # | Architecture | Residual Evaluation | Input Features |
|---|---|---|---|
| 1 | **Vanilla PINN** | Automatic differentiation (autograd) | Raw (x, y) |
| 2 | **FDM-PINN** | Finite-difference stencils | Raw (x, y) |
| 3 | **Fourier (DH) PINN** | Automatic differentiation | Deterministic Harmonic features |
| 4 | **Fourier-FDM PINN** | Finite-difference stencils | Deterministic Harmonic features |

All architectures use **hard boundary constraint enforcement** via squared-envelope ansatz functions.

---

## 📐 Problem Setup

- **PDE:** Biharmonic plate equation $D\nabla^4 w = q(x,y)$
- **Domain:** Unit square $[0,1]^2$
- **Boundary Conditions:** SSSS (Simply Supported) and FFFF (Fully Clamped)
- **Loading Types:** 5 cases — Sinusoidal, Uniform, Hydrostatic, Triangular Prism, Partial Patch

---

## 📁 Repository Structure

```
├── PINNs for Plates (Simply supported all edges)/
│   ├── 1 Venilla PINN/
│   │   ├── 1 sinusoidal loading/
│   │   ├── 2 uniform loading/
│   │   ├── 3 hydrostatic loading/
│   │   ├── 4 triangular prism loading/
│   │   └── 5 patch loading/
│   ├── 2 FDM PINN/
│   ├── 3 Fourier Vanilla PINN/
│   └── 4 Fourier FDM PINN/
│
├── PINNs for Plates (Fixed supported all edges)/
│   ├── Architecture 1 Vanilla PINN/
│   │   ├── 1 sinusoidal loading/
│   │   ├── 2 uniform loading/
│   │   ├── 3 hydrostatic loading/
│   │   ├── 4 triangular loading/
│   │   └── 5 partially loading/
│   ├── Architecture 2 FDM PINN/
│   ├── Architecture 3 Fourier Vanilla PINN/
│   └── Architecture 4 Fourier FDM PINN/
│
├── scripts/
│   ├── generate_all_figures.py    # Reproduce all paper figures
│   └── generate_comparative_figures.py
│
└── README.md
```

Each leaf folder contains:
- **`.ipynb`** — Jupyter notebook with full training code + results
- **`results/error_metrics.json`** — Deflection and moment error metrics
- **`results/cost_metrics.json`** — Training time, GPU memory, parameter count
- **`results/loss_balance.json`** — Training loss history summary

---

## ⚙️ Requirements

```bash
pip install torch numpy matplotlib scipy pandas jupyter
```

**Tested with:**
- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

---

## 🚀 How to Run

### 1. Interactive Execution (Jupyter)

1. Clone the repository:
```bash
git clone https://github.com/vivekkumaryadavrsciv23-source/Efficient-PINN-Kirchhoff-Plate-Bending.git
cd Efficient-PINN-Kirchhoff-Plate-Bending
```

2. Open any benchmark notebook, for example:
```bash
jupyter notebook "PINNs for Plates (Simply supported all edges)/1 Venilla PINN/1 sinusoidal loading/Vanilla-PINN SINUSOIDAL loading.ipynb"
```

3. Run all cells (`Kernel → Restart & Run All`).

---

### 2. Command-Line Execution (Headless / Terminal / Server)

If you prefer running from the command line or on a remote GPU cluster without a browser, you can convert and run any notebook directly as a Python script:

```bash
# Convert notebook to a standard Python (.py) script:
jupyter nbconvert --to script "path/to/notebook.ipynb"
python "path/to/notebook.py"

# Or execute headlessly in terminal:
jupyter nbconvert --to notebook --execute "path/to/notebook.ipynb"
```

---

### 3. Reproducing Manuscript Figures

To reproduce all comparative figures from the benchmark results:
```bash
cd scripts
python generate_all_figures.py
```

---

## 📊 Results Summary

### Deflection Accuracy (Relative L² Error)

| Architecture | SSSS Best | SSSS Worst | FFFF Best | FFFF Worst |
|---|---|---|---|---|
| Vanilla PINN | Sinusoidal | Patch | Sinusoidal | Patch |
| FDM-PINN | ≈ Vanilla | ≈ Vanilla | ≈ Vanilla | ≈ Vanilla |
| Fourier PINN | Improved | Improved | Improved | Improved |
| **Fourier-FDM PINN** | **Best** | **Best** | **Best** | **Best** |

> Full quantitative results are in each notebook's `results/error_metrics.json`

### Computational Efficiency

| Metric | FDM-PINN vs Vanilla |
|---|---|
| Wall-clock speedup | **13×–56×** |
| GPU memory reduction | **up to 37×** |
| Parameters | Same (~8,578) |

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{yadav_pinn_plate,
  title   = {Comparative Assessment of Physics-Informed Neural Network Architectures 
             with Hard Boundary Constraints for Kirchhoff Plate Bending},
  author  = {Yadav, Vivek Kumar and Yadav, Kshitij Kumar and Kumar, Rajesh}
}
```

---

## 📬 Contact

**Vivek Kumar Yadav** — vivekkumaryadav.rs.civ23@iitbhu.ac.in  
Department of Civil Engineering, IIT (BHU), Varanasi — 221005, India
