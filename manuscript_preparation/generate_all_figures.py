"""
Generate ALL publication-quality figures for the CMAME manuscript.
Reads field_data.csv and training_convergence.csv from each experiment
to create properly formatted, high-resolution figures with clear labels.
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

# ═══════════════════════════════════════════════════════════════
# GLOBAL FIGURE STYLING — Publication quality
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.pad_inches': 0.15,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
})
SAVEKW = {'dpi': 300, 'bbox_inches': 'tight', 'pad_inches': 0.12}

BASE = r"d:\# VIVEK\Presentation\research paper\Paper 1 version 2"
SSSS_DIR = os.path.join(BASE, "PINNs for Plates (Simply supported all edges)")
FFFF_DIR = os.path.join(BASE, "PINNs for Plates (Fixed supported all edges)")
OUT_BASE = os.path.join(BASE, "manuscript_preparation", "figures")

# Architecture info
ARCH_INFO = {
    'SSSS': {
        'Vanilla': {'dir': '1 Venilla PINN', 'label': 'Vanilla PINN', 'color': '#1565C0'},
        'FDM': {'dir': '2 FDM PINN', 'label': 'FDM-PINN', 'color': '#2E7D32'},
        'Fourier': {'dir': '3 Fourier Vanilla PINN', 'label': 'Fourier PINN', 'color': '#E65100'},
        'FourierFDM': {'dir': '4 Fourier FDM PINN', 'label': 'Fourier-FDM PINN', 'color': '#AD1457'},
    },
    'FFFF': {
        'Vanilla': {'dir': 'Architecture 1 Vanilla PINN', 'label': 'Vanilla PINN', 'color': '#1565C0'},
        'FDM': {'dir': 'Architecture 2 FDM PINN', 'label': 'FDM-PINN', 'color': '#2E7D32'},
        'Fourier': {'dir': 'Architecture 3 Fourier Vanilla PINN', 'label': 'Fourier PINN', 'color': '#E65100'},
        'FourierFDM': {'dir': 'Architecture 4 Fourier FDM PINN', 'label': 'Fourier-FDM PINN', 'color': '#AD1457'},
    }
}

LOAD_DIRS = {
    'sinusoidal': '1 sinusoidal*',
    'uniform': '2 uniform*',
    'hydrostatic': '3 hydro*',
    'triangular': '4 triangular*',
    'patch': '5*',
}
# FFFF uses slightly different naming
LOAD_DIRS_FFFF = {
    'sinusoidal': '1 sinusoidal*',
    'uniform': '2 uniform*',
    'hydrostatic': '3 hydrostatic*',
    'triangular': '4*triangular*',
    'patch': '5*',
}

LOAD_TITLES = {
    'sinusoidal': 'Sinusoidal Loading',
    'uniform': 'Uniform Loading',
    'hydrostatic': 'Hydrostatic Loading',
    'triangular': 'Triangular Loading',
    'patch': 'Partial Patch Loading',
}

def find_results_dir(bc_base, arch_dir, load_pattern):
    """Find the results directory containing field_data.csv"""
    bc_path = SSSS_DIR if bc_base == 'SSSS' else FFFF_DIR
    arch_path = os.path.join(bc_path, arch_dir)
    load_dirs = glob.glob(os.path.join(arch_path, load_pattern))
    if not load_dirs:
        return None
    for ld in load_dirs:
        results = glob.glob(os.path.join(ld, 'results', '*'))
        for r in results:
            bname = os.path.basename(r)
            if bc_base == 'SSSS' and ('_FFFF' in bname or 'FFFF_' in bname):
                continue
            if bc_base == 'FFFF' and ('_SSSS' in bname or 'SSSS_' in bname):
                continue
            csv_path = os.path.join(r, 'field_data.csv')
            if os.path.isfile(csv_path):
                return r
    return None

def load_field_data(results_dir):
    """Load field_data.csv and return structured data"""
    csv_path = os.path.join(results_dir, 'field_data.csv')
    if not os.path.isfile(csv_path):
        return None
    data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=float)
    return data

def load_convergence(results_dir):
    """Load training_convergence.csv"""
    csv_path = os.path.join(results_dir, 'training_convergence.csv')
    if not os.path.isfile(csv_path):
        return None
    data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=float, max_rows=15001)
    return data

def make_contour(ax, x, y, z, title, cmap='RdYlBu_r', fmt='%.2e', is_2x2=False):
    """Create a publication-quality contour plot with large, bold, legible fonts"""
    n = int(np.sqrt(len(x)))
    if n*n != len(x):
        n = int(round(np.sqrt(len(x))))
    try:
        X = x.reshape(n, n)
        Y = y.reshape(n, n)
        Z = z.reshape(n, n)
    except:
        from scipy.interpolate import griddata
        xi = np.linspace(x.min(), x.max(), 51)
        yi = np.linspace(y.min(), y.max(), 51)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((x, y), z, (X, Y), method='cubic')
    
    cf = ax.contourf(X, Y, Z, levels=20, cmap=cmap)
    
    t_size = 17.5 if not is_2x2 else 17
    lbl_size = 19 if not is_2x2 else 18
    tk_size = 16 if not is_2x2 else 15
    cb_size = 15 if not is_2x2 else 14
    
    cb = plt.colorbar(cf, ax=ax, format=fmt, shrink=0.84, pad=0.04)
    cb.ax.tick_params(labelsize=cb_size, width=1.5, length=5)
    cb.ax.yaxis.get_offset_text().set_fontsize(cb_size)
    cb.ax.yaxis.get_offset_text().set_fontweight('bold')
    
    ax.set_xlabel(r'$x/a$', fontsize=lbl_size, fontweight='bold', labelpad=4)
    ax.set_ylabel(r'$y/b$', fontsize=lbl_size, fontweight='bold', labelpad=4)
    ax.set_title(title, fontsize=t_size, fontweight='bold', pad=12)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=tk_size, width=1.5, length=5)
    return cf


def generate_summary_figure(bc_type, load_type, arch_key, out_dir):
    """Generate a 2x3 summary figure: w_ref, w_pred, |error|, Mx, My, Mxy"""
    bc_base = SSSS_DIR if bc_type == 'SSSS' else FFFF_DIR
    info = ARCH_INFO[bc_type][arch_key]
    load_pat = LOAD_DIRS[load_type] if bc_type == 'SSSS' else LOAD_DIRS_FFFF[load_type]
    
    rdir = find_results_dir(bc_type, info['dir'], load_pat)
    if rdir is None:
        print(f"  SKIP: {bc_type}/{arch_key}/{load_type} - no results dir")
        return False
    
    data = load_field_data(rdir)
    if data is None:
        print(f"  SKIP: {bc_type}/{arch_key}/{load_type} - no field_data.csv")
        return False
    
    x = data['x_nd']
    y = data['y_nd']
    
    # Get column names - they vary slightly between experiments
    col_names = data.dtype.names
    
    # Find reference and predicted deflection columns
    w_ref_col = 'w_dim_ref' if 'w_dim_ref' in col_names else 'w_dim_anal' if 'w_dim_anal' in col_names else None
    w_pred_col = 'w_dim_pred'
    
    if w_ref_col is None:
        for c in col_names:
            if 'ref' in c.lower() and 'w' in c.lower() and 'error' not in c.lower():
                w_ref_col = c
                break
            elif 'anal' in c.lower() and 'w' in c.lower():
                w_ref_col = c
                break
    
    if w_ref_col is None or w_pred_col not in col_names:
        print(f"  SKIP: {bc_type}/{arch_key}/{load_type} - missing w columns: {col_names}")
        return False
    
    w_ref = data[w_ref_col]
    w_pred = data[w_pred_col]
    w_err = np.abs(w_pred - w_ref)
    
    # Moment columns
    Mx_pred = data['Mx_dim_pred'] if 'Mx_dim_pred' in col_names else None
    My_pred = data['My_dim_pred'] if 'My_dim_pred' in col_names else None
    Mxy_pred = data['Mxy_dim_pred'] if 'Mxy_dim_pred' in col_names else None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
    fig.suptitle(f'{info["label"]} — {LOAD_TITLES[load_type]} ({bc_type})',
                 fontsize=26, fontweight='bold', y=0.98)
    
    w_max = np.max(np.abs(w_ref * 1e3))
    w_fmt = '%.4f' if w_max < 0.1 else '%.3f'
    
    # Row 1: w_ref, w_pred, |error|
    make_contour(axes[0,0], x, y, w_ref*1e3, 
                 r'(a) Reference $w_{\mathrm{ref}}$ [mm]', 'RdYlBu_r', w_fmt)
    make_contour(axes[0,1], x, y, w_pred*1e3, 
                 r'(b) Predicted $w_{\mathrm{pred}}$ [mm]', 'RdYlBu_r', w_fmt)
    make_contour(axes[0,2], x, y, w_err, 
                 r'(c) Abs. Error $|w_{\mathrm{pred}} - w_{\mathrm{ref}}|$ [m]', 'hot_r', '%.1e')
    
    # Row 2: Mx, My, Mxy
    if Mx_pred is not None:
        make_contour(axes[1,0], x, y, Mx_pred, 
                     r'(d) Bending Moment $M_x$ [N$\cdot$m/m]', 'coolwarm', '%.1f')
    if My_pred is not None:
        make_contour(axes[1,1], x, y, My_pred, 
                     r'(e) Bending Moment $M_y$ [N$\cdot$m/m]', 'coolwarm', '%.1f')
    if Mxy_pred is not None:
        make_contour(axes[1,2], x, y, Mxy_pred, 
                     r'(f) Twisting Moment $M_{xy}$ [N$\cdot$m/m]', 'coolwarm', '%.1f')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=2.0)
    
    fname = f'{bc_type}_{arch_key}_{load_type}_summary.png'
    plt.savefig(os.path.join(out_dir, fname), **SAVEKW)
    plt.close()
    return True


def generate_convergence_figure(bc_type, load_type, out_dir):
    """Generate training loss comparison for all 4 architectures in one figure"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Training Loss Convergence — {LOAD_TITLES[load_type]} ({bc_type})',
                 fontsize=26, fontweight='bold', y=0.98)
    
    arch_keys = ['Vanilla', 'FDM', 'Fourier', 'FourierFDM']
    labels = ['(a) Vanilla PINN', '(b) FDM-PINN', '(c) Fourier PINN', '(d) Fourier-FDM PINN']
    
    for idx, (ak, lbl) in enumerate(zip(arch_keys, labels)):
        ax = axes[idx//2, idx%2]
        info = ARCH_INFO[bc_type][ak]
        load_pat = LOAD_DIRS[load_type] if bc_type == 'SSSS' else LOAD_DIRS_FFFF[load_type]
        
        rdir = find_results_dir(bc_type, info['dir'], load_pat)
        if rdir is None:
            ax.text(0.5, 0.5, 'Data not found', transform=ax.transAxes, ha='center', fontsize=18, fontweight='bold')
            ax.set_title(lbl, fontsize=19, fontweight='bold')
            continue
        
        conv = load_convergence(rdir)
        if conv is None:
            ax.text(0.5, 0.5, 'No convergence data', transform=ax.transAxes, ha='center', fontsize=18, fontweight='bold')
            ax.set_title(lbl, fontsize=19, fontweight='bold')
            continue
        
        epochs = conv['epoch']
        total_loss = conv['total_loss']
        pde_loss = conv['pde_loss']
        
        ax.semilogy(epochs, total_loss, color=info['color'], linewidth=2.5, label='Total Loss', alpha=0.95)
        if not np.allclose(total_loss, pde_loss):
            ax.semilogy(epochs, pde_loss, color=info['color'], linewidth=2.0, linestyle='--', label='PDE Loss', alpha=0.75)
        
        ax.set_xlabel('Epoch', fontsize=18, fontweight='bold', labelpad=6)
        ax.set_ylabel('Loss (log scale)', fontsize=18, fontweight='bold', labelpad=6)
        ax.set_title(lbl, fontsize=19, fontweight='bold', pad=10)
        ax.tick_params(labelsize=15, width=1.5, length=5)
        ax.legend(fontsize=15, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.35, linewidth=0.8)
        ax.set_xlim([0, epochs[-1]])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=2.0)
    fname = f'{bc_type}_{load_type}_convergence.png'
    plt.savefig(os.path.join(out_dir, fname), **SAVEKW)
    plt.close()
    return True


def generate_error_contour_comparison(bc_type, load_type, out_dir):
    """Generate absolute error comparison: 4 panels, one per architecture"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle(f'Deflection Absolute Error — {LOAD_TITLES[load_type]} ({bc_type})',
                 fontsize=26, fontweight='bold', y=0.98)
    
    arch_keys = ['Vanilla', 'FDM', 'Fourier', 'FourierFDM']
    labels = ['(a) Vanilla PINN', '(b) FDM-PINN', '(c) Fourier PINN', '(d) Fourier-FDM PINN']
    
    # First pass: find global max error for consistent colorbar
    all_errors = []
    for ak in arch_keys:
        info = ARCH_INFO[bc_type][ak]
        load_pat = LOAD_DIRS[load_type] if bc_type == 'SSSS' else LOAD_DIRS_FFFF[load_type]
        rdir = find_results_dir(bc_type, info['dir'], load_pat)
        if rdir:
            data = load_field_data(rdir)
            if data is not None:
                cols = data.dtype.names
                w_ref_col = 'w_dim_ref' if 'w_dim_ref' in cols else 'w_dim_anal' if 'w_dim_anal' in cols else None
                if w_ref_col is None:
                    for c in cols:
                        if ('ref' in c.lower() or 'anal' in c.lower()) and 'w' in c.lower() and 'error' not in c.lower():
                            w_ref_col = c; break
                if w_ref_col:
                    err = np.abs(data['w_dim_pred'] - data[w_ref_col])
                    all_errors.append(err.max())
    
    vmax = max(all_errors) if all_errors else 1e-6
    
    for idx, (ak, lbl) in enumerate(zip(arch_keys, labels)):
        ax = axes[idx//2, idx%2]
        info = ARCH_INFO[bc_type][ak]
        load_pat = LOAD_DIRS[load_type] if bc_type == 'SSSS' else LOAD_DIRS_FFFF[load_type]
        rdir = find_results_dir(bc_type, info['dir'], load_pat)
        
        if rdir is None:
            ax.text(0.5, 0.5, 'Data not found', transform=ax.transAxes, ha='center', fontsize=18, fontweight='bold')
            ax.set_title(lbl, fontsize=19, fontweight='bold')
            continue
        
        data = load_field_data(rdir)
        if data is None:
            continue
        
        x, y = data['x_nd'], data['y_nd']
        cols = data.dtype.names
        w_ref_col = 'w_dim_ref' if 'w_dim_ref' in cols else 'w_dim_anal' if 'w_dim_anal' in cols else None
        if w_ref_col is None:
            for c in cols:
                if ('ref' in c.lower() or 'anal' in c.lower()) and 'w' in c.lower() and 'error' not in c.lower():
                    w_ref_col = c; break
        
        if w_ref_col:
            w_err = np.abs(data['w_dim_pred'] - data[w_ref_col])
            make_contour(ax, x, y, w_err, lbl, 'hot_r', '%.1e', is_2x2=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=2.0)
    fname = f'{bc_type}_{load_type}_error_comparison.png'
    plt.savefig(os.path.join(out_dir, fname), **SAVEKW)
    plt.close()
    return True


# ═══════════════════════════════════════════════════════════════
# COMPARATIVE FIGURES (from numerical data)
# ═══════════════════════════════════════════════════════════════

def generate_comparative_figures(out_dir):
    """Generate bar charts, heatmaps, Pareto plots from the numerical results"""
    
    archs = ['Vanilla PINN', 'FDM-PINN', 'Fourier PINN', 'Fourier-FDM']
    arch_colors = ['#1565C0', '#2E7D32', '#E65100', '#AD1457']
    loads = ['Sinusoidal', 'Uniform', 'Hydrostatic', 'Triangular', 'Patch']
    x = np.arange(len(loads))
    w = 0.19

    # ─── DATA ───
    ssss_center = np.array([
        [0.131, 0.264, 0.040, 0.110],
        [0.106, 0.149, 0.289, 0.068],
        [0.246, 0.263, 0.617, 0.732],
        [1.485, 0.435, 0.269, 0.264],
        [0.143, 0.801, 0.251, 0.395],
    ])
    ffff_center = np.array([
        [0.370, 0.025, 0.323, 0.328],
        [0.187, 0.004, 0.336, 0.345],
        [0.422, 0.261, 0.063, 0.143],
        [0.372, 0.097, 0.330, 0.020],
        [1.802, 1.510, 0.145, 0.015],
    ])
    ssss_rel_l2 = np.array([
        [1.51e-3, 2.97e-3, 6.32e-4, 1.26e-3],
        [2.65e-3, 2.99e-3, 3.22e-3, 1.04e-3],
        [1.38e-2, 1.38e-2, 2.35e-2, 2.58e-2],
        [1.70e-2, 5.77e-3, 3.45e-3, 3.33e-3],
        [8.48e-3, 1.26e-2, 1.05e-2, 1.30e-2],
    ])
    ffff_rel_l2 = np.array([
        [4.66e-3, 2.64e-4, 4.21e-3, 4.25e-3],
        [2.19e-3, 6.93e-5, 4.79e-3, 4.86e-3],
        [1.11e-1, 1.12e-1, 7.58e-4, 1.28e-3],
        [4.93e-3, 1.17e-3, 4.86e-3, 2.64e-4],
        [1.85e-2, 2.52e-2, 1.80e-2, 1.79e-2],
    ])
    ffff_mx_rms = np.array([
        [2.589, 0.175, 2.591, 0.021],
        [3.793, 0.364, 4.438, 0.035],
        [2.154, 2.159, 0.202, 0.132],
        [2.490, 1.803, 2.349, 0.103],
        [1.258, 0.271, 0.237, 0.245],
    ])
    ssss_time = np.array([
        [2583, 194, 5350, 461],
        [2580, 169, 5915, 451],
        [6934, 177, 6001, 385],
        [6784, 608, 3155, 446],
        [2952, 558, 13102, 1194],
    ])
    ffff_time = np.array([
        [8384, 212, 16132, 227],
        [8345, 254, 16143, 181],
        [8065, 187, 7158, 1147],
        [6832, 517, 16149, 1016],
        [6835, 121, 4948, 606],
    ])
    ssss_gpu = np.array([
        [1110, 30, 1171, 132],
        [1110, 30, 1182, 138],
        [1964, 30, 1221, 259],
        [1110, 90, 1884, 135],
        [1118, 29, 1182, 141],
    ])
    ffff_gpu = np.array([
        [1117, 30, 1220, 30],
        [1117, 30, 1220, 30],
        [1131, 30, 1232, 139],
        [1117, 89, 1220, 135],
        [1117, 36, 1198, 135],
    ])

    # ─── FIG 1: Center error bar chart (SSSS + FFFF side by side) ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    for i, (a, c) in enumerate(zip(archs, arch_colors)):
        ax1.bar(x + i*w - 1.5*w, ssss_center[:, i], w, label=a, color=c, edgecolor='white', lw=0.8)
        ax2.bar(x + i*w - 1.5*w, ffff_center[:, i], w, label=a, color=c, edgecolor='white', lw=0.8)
    for ax, title in [(ax1, 'Simply Supported (SSSS)'), (ax2, 'Fully Clamped (FFFF)')]:
        ax.set_xlabel('Loading Case', fontsize=18, fontweight='bold', labelpad=6)
        ax.set_ylabel('Center Deflection Error (%)', fontsize=18, fontweight='bold', labelpad=6)
        ax.set_title(title, fontsize=19, fontweight='bold', pad=10)
        ax.set_xticks(x); ax.set_xticklabels(loads, fontsize=15, fontweight='bold')
        ax.set_yscale('log'); ax.legend(fontsize=14); ax.grid(axis='y', alpha=0.35)
        ax.tick_params(labelsize=15, width=1.5, length=5)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(out_dir, 'center_error_comparison.png'), **SAVEKW)
    plt.close()
    print("  Comparative 1/8: center_error_comparison")

    # ─── FIG 2: L2 error heatmaps ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    for ax, data, title in [(ax1, ssss_rel_l2, 'SSSS'), (ax2, ffff_rel_l2, 'FFFF')]:
        im = ax.imshow(np.log10(data), cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(4)); ax.set_xticklabels(['Vanilla', 'FDM', 'Fourier', 'F-FDM'], fontsize=16, fontweight='bold')
        ax.set_yticks(range(5)); ax.set_yticklabels(loads, fontsize=16, fontweight='bold')
        for i in range(5):
            for j in range(4):
                ax.text(j, i, f'{data[i,j]:.1e}', ha='center', va='center', fontsize=14,
                        color='white' if np.log10(data[i,j]) > -1.8 else 'black', fontweight='bold')
        ax.set_title(f'Relative $L^2$ Error — {title}', fontsize=19, fontweight='bold', pad=10)
        cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
        cb.set_label(r'$\log_{10}(\epsilon_{L^2})$', fontsize=16, fontweight='bold')
        cb.ax.tick_params(labelsize=14, width=1.5, length=5)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(out_dir, 'L2_error_heatmaps.png'), **SAVEKW)
    plt.close()
    print("  Comparative 2/8: L2_error_heatmaps")

    # ─── FIG 3: Mx RMS bar chart (FFFF) ───
    fig, ax = plt.subplots(figsize=(12, 7.5))
    for i, (a, c) in enumerate(zip(archs, arch_colors)):
        ax.bar(x + i*w - 1.5*w, ffff_mx_rms[:, i], w, label=a, color=c, edgecolor='white', lw=0.8)
    ax.set_xlabel('Loading Case', fontsize=18, fontweight='bold', labelpad=6)
    ax.set_ylabel(r'$M_x$ RMS Error (N$\cdot$m)', fontsize=18, fontweight='bold', labelpad=6)
    ax.set_title(r'Bending Moment $M_x$ RMS Error — FFFF', fontsize=20, fontweight='bold', pad=10)
    ax.set_xticks(x); ax.set_xticklabels(loads, fontsize=15, fontweight='bold')
    ax.set_yscale('log'); ax.legend(fontsize=15); ax.grid(axis='y', alpha=0.35)
    ax.tick_params(labelsize=15, width=1.5, length=5)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(out_dir, 'FFFF_Mx_RMS_comparison.png'), **SAVEKW)
    plt.close()
    print("  Comparative 3/8: FFFF_Mx_RMS_comparison")

    # ─── FIG 4: Training time bar chart ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    for i, (a, c) in enumerate(zip(archs, arch_colors)):
        ax1.bar(x + i*w - 1.5*w, ssss_time[:, i]/60, w, label=a, color=c, edgecolor='white', lw=0.8)
        ax2.bar(x + i*w - 1.5*w, ffff_time[:, i]/60, w, label=a, color=c, edgecolor='white', lw=0.8)
    for ax, title in [(ax1, 'SSSS'), (ax2, 'FFFF')]:
        ax.set_xlabel('Loading Case', fontsize=18, fontweight='bold', labelpad=6)
        ax.set_ylabel('Training Time (minutes)', fontsize=18, fontweight='bold', labelpad=6)
        ax.set_title(f'Training Time — {title}', fontsize=19, fontweight='bold', pad=10)
        ax.set_xticks(x); ax.set_xticklabels(loads, fontsize=15, fontweight='bold')
        ax.set_yscale('log'); ax.legend(fontsize=14); ax.grid(axis='y', alpha=0.35)
        ax.tick_params(labelsize=15, width=1.5, length=5)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(out_dir, 'training_time_comparison.png'), **SAVEKW)
    plt.close()
    print("  Comparative 4/8: training_time_comparison")

    # ─── FIG 5: GPU memory bar chart ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    for i, (a, c) in enumerate(zip(archs, arch_colors)):
        ax1.bar(x + i*w - 1.5*w, ssss_gpu[:, i], w, label=a, color=c, edgecolor='white', lw=0.8)
        ax2.bar(x + i*w - 1.5*w, ffff_gpu[:, i], w, label=a, color=c, edgecolor='white', lw=0.8)
    for ax, title in [(ax1, 'SSSS'), (ax2, 'FFFF')]:
        ax.set_xlabel('Loading Case', fontsize=18, fontweight='bold', labelpad=6)
        ax.set_ylabel('Peak GPU Memory (MB)', fontsize=18, fontweight='bold', labelpad=6)
        ax.set_title(f'GPU Memory — {title}', fontsize=19, fontweight='bold', pad=10)
        ax.set_xticks(x); ax.set_xticklabels(loads, fontsize=15, fontweight='bold')
        ax.set_yscale('log'); ax.legend(fontsize=14); ax.grid(axis='y', alpha=0.35)
        ax.tick_params(labelsize=15, width=1.5, length=5)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(out_dir, 'GPU_memory_comparison.png'), **SAVEKW)
    plt.close()
    print("  Comparative 5/8: GPU_memory_comparison")

    # ─── FIG 6: Speedup & Memory Reduction ───
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))
    sp_ssss = ssss_time[:, 0] / ssss_time[:, 1]
    sp_ffff = ffff_time[:, 0] / ffff_time[:, 1]
    mr_ssss = ssss_gpu[:, 0] / ssss_gpu[:, 1]
    mr_ffff = ffff_gpu[:, 0] / ffff_gpu[:, 1]
    
    ax1.bar(x - 0.15, sp_ssss, 0.3, label='SSSS', color='#1565C0', alpha=0.85)
    ax1.bar(x + 0.15, sp_ffff, 0.3, label='FFFF', color='#AD1457', alpha=0.85)
    ax1.set_xlabel('Loading Case', fontsize=18, fontweight='bold', labelpad=6)
    ax1.set_ylabel('Speedup Factor (×)', fontsize=18, fontweight='bold', labelpad=6)
    ax1.set_title('FDM-PINN Speedup over Vanilla', fontsize=19, fontweight='bold', pad=10)
    ax1.set_xticks(x); ax1.set_xticklabels(loads, rotation=15, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=15); ax1.grid(axis='y', alpha=0.35); ax1.tick_params(labelsize=15, width=1.5, length=5)
    for i in range(5):
        ax1.text(i-0.15, sp_ssss[i]+0.8, f'{sp_ssss[i]:.0f}×', ha='center', fontsize=14, fontweight='bold')
        ax1.text(i+0.15, sp_ffff[i]+0.8, f'{sp_ffff[i]:.0f}×', ha='center', fontsize=14, fontweight='bold')

    ax2.bar(x - 0.15, mr_ssss, 0.3, label='SSSS', color='#1565C0', alpha=0.85)
    ax2.bar(x + 0.15, mr_ffff, 0.3, label='FFFF', color='#AD1457', alpha=0.85)
    ax2.set_xlabel('Loading Case', fontsize=18, fontweight='bold', labelpad=6)
    ax2.set_ylabel('Memory Reduction (×)', fontsize=18, fontweight='bold', labelpad=6)
    ax2.set_title('GPU Memory: FDM vs Vanilla', fontsize=19, fontweight='bold', pad=10)
    ax2.set_xticks(x); ax2.set_xticklabels(loads, rotation=15, fontsize=14, fontweight='bold')
    ax2.legend(fontsize=15); ax2.grid(axis='y', alpha=0.35); ax2.tick_params(labelsize=15, width=1.5, length=5)
    for i in range(5):
        ax2.text(i-0.15, mr_ssss[i]+0.5, f'{mr_ssss[i]:.0f}×', ha='center', fontsize=14, fontweight='bold')
        ax2.text(i+0.15, mr_ffff[i]+0.5, f'{mr_ffff[i]:.0f}×', ha='center', fontsize=14, fontweight='bold')
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(out_dir, 'speedup_memory_reduction.png'), **SAVEKW)
    plt.close()
    print("  Comparative 6/8: speedup_memory_reduction")

    # ─── FIG 7: Pareto front ───
    fig, ax = plt.subplots(figsize=(13, 8.5))
    markers = ['o', 's', '^', 'D', 'P']
    for bc_label, rl2, t in [('SSSS', ssss_rel_l2, ssss_time), ('FFFF', ffff_rel_l2, ffff_time)]:
        for i, (a, c) in enumerate(zip(archs, arch_colors)):
            for j, (l, m) in enumerate(zip(loads, markers)):
                lbl = a if (j == 0 and bc_label == 'SSSS') else None
                fc = c if bc_label == 'SSSS' else 'none'
                ax.scatter(t[j,i]/60, rl2[j,i], c=fc, edgecolors=c, marker=m, s=110, alpha=0.85,
                          label=lbl, linewidths=2.2, facecolors=fc)
    ax.set_xlabel('Training Time (minutes)', fontsize=18, fontweight='bold', labelpad=6)
    ax.set_ylabel(r'Relative $L^2$ Error', fontsize=18, fontweight='bold', labelpad=6)
    ax.set_title('Accuracy vs Cost — All 40 Cases\n(Filled = SSSS, Open = FFFF)', fontsize=20, fontweight='bold', pad=10)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, alpha=0.35)
    ax.tick_params(labelsize=15, width=1.5, length=5)
    h1 = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=12, label=a) 
          for a, c in zip(archs, arch_colors)]
    h2 = [plt.Line2D([0],[0], marker=m, color='gray', markersize=12, label=l, linestyle='None')
          for l, m in zip(loads, markers)]
    l1 = ax.legend(handles=h1, loc='upper right', title='Architecture', fontsize=13, title_fontsize=14)
    ax.add_artist(l1)
    ax.legend(handles=h2, loc='lower left', title='Loading', fontsize=13, title_fontsize=14)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(out_dir, 'Pareto_accuracy_vs_cost.png'), **SAVEKW)
    plt.close()
    print("  Comparative 7/8: Pareto_accuracy_vs_cost")

    # ─── FIG 8: Mx heatmap (FFFF) ───
    fig, ax = plt.subplots(figsize=(11, 7.5))
    im = ax.imshow(np.log10(ffff_mx_rms), cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels(['Vanilla', 'FDM', 'Fourier', 'F-FDM'], fontsize=16, fontweight='bold')
    ax.set_yticks(range(5)); ax.set_yticklabels(loads, fontsize=16, fontweight='bold')
    for i in range(5):
        for j in range(4):
            ax.text(j, i, f'{ffff_mx_rms[i,j]:.3f}', ha='center', va='center', fontsize=15,
                    color='white' if np.log10(ffff_mx_rms[i,j]) > 0 else 'black', fontweight='bold')
    ax.set_title(r'$M_x$ RMS Error (N$\cdot$m) — FFFF', fontsize=20, fontweight='bold', pad=10)
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cb.set_label(r'$\log_{10}(M_x$ RMS$)$', fontsize=16, fontweight='bold')
    cb.ax.tick_params(labelsize=14, width=1.5, length=5)
    ax.tick_params(labelsize=15, width=1.5, length=5)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(out_dir, 'FFFF_Mx_heatmap.png'), **SAVEKW)
    plt.close()
    print("  Comparative 8/8: FFFF_Mx_heatmap")


# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("="*60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*60)
    
    comp_dir = os.path.join(OUT_BASE, 'comparative')
    main_dir = os.path.join(OUT_BASE, 'main_body')
    app_dir = os.path.join(OUT_BASE, 'appendix')
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(app_dir, exist_ok=True)
    
    # ─── MAIN BODY: sinusoidal only ───
    print("\n--- Main Body Figures (Sinusoidal Only) ---")
    for bc in ['SSSS', 'FFFF']:
        for ak in ['Vanilla', 'FDM', 'Fourier', 'FourierFDM']:
            ok = generate_summary_figure(bc, 'sinusoidal', ak, main_dir)
            print(f"  {'OK' if ok else 'SKIP'}: {bc}/{ak}/sinusoidal summary")
    
    # Convergence and error comparison for sinusoidal
    for bc in ['SSSS', 'FFFF']:
        generate_convergence_figure(bc, 'sinusoidal', main_dir)
        print(f"  OK: {bc}/sinusoidal convergence")
        generate_error_contour_comparison(bc, 'sinusoidal', main_dir)
        print(f"  OK: {bc}/sinusoidal error comparison")
    
    # ─── APPENDIX: remaining loading cases ───
    print("\n--- Appendix Figures (Remaining Loadings) ---")
    for load in ['uniform', 'hydrostatic', 'triangular', 'patch']:
        for bc in ['SSSS', 'FFFF']:
            for ak in ['Vanilla', 'FDM', 'Fourier', 'FourierFDM']:
                ok = generate_summary_figure(bc, load, ak, app_dir)
                if ok: print(f"  OK: {bc}/{ak}/{load} summary")
            generate_convergence_figure(bc, load, app_dir)
            print(f"  OK: {bc}/{load} convergence")
            generate_error_contour_comparison(bc, load, app_dir)
            print(f"  OK: {bc}/{load} error comparison")
    
    # ─── COMPARATIVE ───
    print("\n--- Comparative Figures ---")
    generate_comparative_figures(comp_dir)
    
    # ─── Count ───
    total_main = len([f for f in os.listdir(main_dir) if f.endswith('.png')])
    total_app = len([f for f in os.listdir(app_dir) if f.endswith('.png')])
    total_comp = len([f for f in os.listdir(comp_dir) if f.endswith('.png')])
    print(f"\n{'='*60}")
    print(f"DONE! Generated figures:")
    print(f"  Main body:    {total_main} figures in {main_dir}")
    print(f"  Appendix:     {total_app} figures in {app_dir}")
    print(f"  Comparative:  {total_comp} figures in {comp_dir}")
    print(f"  TOTAL:        {total_main + total_app + total_comp} new figures")
    print(f"{'='*60}")
