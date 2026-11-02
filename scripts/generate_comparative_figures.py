"""
Generate comparative figures for the CMAME manuscript.
Creates bar charts, heatmaps, and Pareto plots comparing all 4 architectures.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox_inches': 'tight',
    'font.family': 'serif'
})

OUT = r"d:\# VIVEK\Presentation\research paper\Paper 1 version 2\manuscript_preparation\figures\comparative"
os.makedirs(OUT, exist_ok=True)

archs = ['Vanilla PINN', 'FDM-PINN', 'Fourier PINN', 'Fourier-FDM PINN']
arch_colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
loads = ['Sinusoidal', 'Uniform', 'Hydrostatic', 'Triangular', 'Patch']

# ═══ SSSS DATA (from error_metrics.json) ═══
ssss_center_err = np.array([
    [0.131, 0.264, 0.040, 0.110],   # sinusoidal
    [0.106, 0.149, 0.289, 0.068],   # uniform
    [0.246, 0.263, 0.617, 0.732],   # hydrostatic
    [1.485, 0.435, 0.269, 0.264],   # triangular
    [0.143, 0.801, 0.251, 0.395],   # patch
])

ssss_rel_l2 = np.array([
    [1.51e-3, 2.97e-3, 6.32e-4, 1.26e-3],
    [2.65e-3, 2.99e-3, 3.22e-3, 1.04e-3],
    [1.38e-2, 1.38e-2, 2.35e-2, 2.58e-2],
    [1.70e-2, 5.77e-3, 3.45e-3, 3.33e-3],
    [8.48e-3, 1.26e-2, 1.05e-2, 1.30e-2],
])

ssss_mx_rms = np.array([
    [0.043, 0.072, 0.046, 0.049],
    [0.659, 0.660, 0.368, 0.241],
    [0.839, 0.840, 1.419, 1.502],
    [0.544, 0.301, 0.247, 0.248],
    [0.302, 0.553, 0.280, 0.328],
])

ssss_time = np.array([
    [2583, 194, 5350, 461],
    [2580, 169, 5915, 451],
    [6934, 177, 6001, 385],
    [6784, 608, 3155, 446],
    [2952, 558, 13102, 1194],
])

ssss_gpu = np.array([
    [1110, 30, 1171, 132],
    [1110, 30, 1182, 138],
    [1964, 30, 1221, 259],
    [1110, 90, 1884, 135],
    [1118, 29, 1182, 141],
])

# ═══ FFFF DATA ═══
ffff_center_err = np.array([
    [0.370, 0.025, 0.323, 0.328],
    [0.187, 0.004, 0.336, 0.345],
    [0.422, 0.261, 0.063, 0.143],
    [0.372, 0.097, 0.330, 0.020],
    [1.802, 1.510, 0.145, 0.015],
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

ffff_time = np.array([
    [8384, 212, 16132, 227],
    [8345, 254, 16143, 181],
    [8065, 187, 7158, 1147],
    [6832, 517, 16149, 1016],
    [6835, 121, 4948, 606],
])

ffff_gpu = np.array([
    [1117, 30, 1220, 30],
    [1117, 30, 1220, 30],
    [1131, 30, 1232, 139],
    [1117, 89, 1220, 135],
    [1117, 36, 1198, 135],
])


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Center Deflection Error — Grouped Bar Chart (SSSS)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(loads))
w = 0.2
for i, (arch, color) in enumerate(zip(archs, arch_colors)):
    ax.bar(x + i*w - 1.5*w, ssss_center_err[:, i], w, label=arch, color=color, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Loading Case')
ax.set_ylabel('Center Deflection Error (%)')
ax.set_title('Center Deflection Error — Simply Supported (SSSS)')
ax.set_xticks(x)
ax.set_xticklabels(loads)
ax.legend(loc='upper left')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'SSSS_center_error_comparison.png'))
plt.close()
print("1/10 done")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Center Deflection Error — Grouped Bar Chart (FFFF)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
for i, (arch, color) in enumerate(zip(archs, arch_colors)):
    ax.bar(x + i*w - 1.5*w, ffff_center_err[:, i], w, label=arch, color=color, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Loading Case')
ax.set_ylabel('Center Deflection Error (%)')
ax.set_title('Center Deflection Error — Fully Clamped (FFFF)')
ax.set_xticks(x)
ax.set_xticklabels(loads)
ax.legend(loc='upper left')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'FFFF_center_error_comparison.png'))
plt.close()
print("2/10 done")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Relative L2 Error Heatmap (SSSS)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(np.log10(ssss_rel_l2), cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(4))
ax.set_xticklabels(['Vanilla', 'FDM', 'Fourier', 'Fourier-FDM'], fontsize=10)
ax.set_yticks(range(5))
ax.set_yticklabels(loads, fontsize=10)
for i in range(5):
    for j in range(4):
        ax.text(j, i, f'{ssss_rel_l2[i,j]:.2e}', ha='center', va='center', fontsize=8,
                color='white' if np.log10(ssss_rel_l2[i,j]) > -1.8 else 'black')
ax.set_title('Relative $L^2$ Error — Simply Supported (SSSS)')
cbar = plt.colorbar(im, ax=ax, label='$\\log_{10}(\\epsilon_{L^2})$')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'SSSS_L2_error_heatmap.png'))
plt.close()
print("3/10 done")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Relative L2 Error Heatmap (FFFF)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(np.log10(ffff_rel_l2), cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(4))
ax.set_xticklabels(['Vanilla', 'FDM', 'Fourier', 'Fourier-FDM'], fontsize=10)
ax.set_yticks(range(5))
ax.set_yticklabels(loads, fontsize=10)
for i in range(5):
    for j in range(4):
        ax.text(j, i, f'{ffff_rel_l2[i,j]:.2e}', ha='center', va='center', fontsize=8,
                color='white' if np.log10(ffff_rel_l2[i,j]) > -1.8 else 'black')
ax.set_title('Relative $L^2$ Error — Fully Clamped (FFFF)')
cbar = plt.colorbar(im, ax=ax, label='$\\log_{10}(\\epsilon_{L^2})$')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'FFFF_L2_error_heatmap.png'))
plt.close()
print("4/10 done")

# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Mx RMS Error Bar Chart (FFFF) — shows Fourier-FDM dominance
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
for i, (arch, color) in enumerate(zip(archs, arch_colors)):
    ax.bar(x + i*w - 1.5*w, ffff_mx_rms[:, i], w, label=arch, color=color, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Loading Case')
ax.set_ylabel('$M_x$ RMS Error (N·m)')
ax.set_title('Bending Moment $M_x$ RMS Error — Fully Clamped (FFFF)')
ax.set_xticks(x)
ax.set_xticklabels(loads)
ax.legend(loc='upper right')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'FFFF_Mx_RMS_comparison.png'))
plt.close()
print("5/10 done")

# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Training Time Bar Chart (FFFF) — shows FDM speedup
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
for i, (arch, color) in enumerate(zip(archs, arch_colors)):
    ax.bar(x + i*w - 1.5*w, ffff_time[:, i]/60, w, label=arch, color=color, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Loading Case')
ax.set_ylabel('Total Training Time (minutes)')
ax.set_title('Training Time — Fully Clamped (FFFF)')
ax.set_xticks(x)
ax.set_xticklabels(loads)
ax.legend(loc='upper left')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'FFFF_training_time_comparison.png'))
plt.close()
print("6/10 done")

# ═══════════════════════════════════════════════════════════════
# FIGURE 7: GPU Memory Bar Chart (FFFF)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
for i, (arch, color) in enumerate(zip(archs, arch_colors)):
    ax.bar(x + i*w - 1.5*w, ffff_gpu[:, i], w, label=arch, color=color, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Loading Case')
ax.set_ylabel('Peak GPU Memory (MB)')
ax.set_title('GPU Memory Consumption — Fully Clamped (FFFF)')
ax.set_xticks(x)
ax.set_xticklabels(loads)
ax.legend(loc='upper right')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'FFFF_GPU_memory_comparison.png'))
plt.close()
print("7/10 done")

# ═══════════════════════════════════════════════════════════════
# FIGURE 8: Pareto Front — Accuracy vs Cost (ALL 40 cases)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
markers = ['o', 's', '^', 'D', 'P']  # per loading
for bc_label, rel_l2, time, marker_offset in [('SSSS', ssss_rel_l2, ssss_time, 0), ('FFFF', ffff_rel_l2, ffff_time, 0)]:
    for i, (arch, color) in enumerate(zip(archs, arch_colors)):
        for j, (load, marker) in enumerate(zip(loads, markers)):
            label = arch if (j == 0 and bc_label == 'SSSS') else None
            edge = 'black' if bc_label == 'FFFF' else color
            fill = color if bc_label == 'SSSS' else 'none'
            ax.scatter(time[j, i]/60, rel_l2[j, i], 
                      c=color if bc_label == 'SSSS' else 'none',
                      edgecolors=color, marker=marker, s=60, alpha=0.8,
                      label=label, linewidths=1.5,
                      facecolors=color if bc_label == 'SSSS' else 'none')

ax.set_xlabel('Total Training Time (minutes)')
ax.set_ylabel('Relative $L^2$ Error')
ax.set_title('Accuracy vs. Computational Cost — All 40 Experiments\n(Filled = SSSS, Open = FFFF)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Architecture legend
arch_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=8, label=a) 
                for a, c in zip(archs, arch_colors)]
load_handles = [plt.Line2D([0], [0], marker=m, color='gray', markersize=8, label=l, linestyle='None')
                for l, m in zip(loads, markers)]
leg1 = ax.legend(handles=arch_handles, loc='upper right', title='Architecture')
ax.add_artist(leg1)
ax.legend(handles=load_handles, loc='lower left', title='Loading', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'Pareto_accuracy_vs_cost.png'))
plt.close()
print("8/10 done")

# ═══════════════════════════════════════════════════════════════
# FIGURE 9: Speedup Ratios (FDM vs Vanilla, Fourier-FDM vs Fourier)
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# FDM vs Vanilla speedup
speedup_fdm = ssss_time[:, 0] / ssss_time[:, 1]  # SSSS
speedup_fdm_ffff = ffff_time[:, 0] / ffff_time[:, 1]  # FFFF

ax1.bar(x - 0.15, speedup_fdm, 0.3, label='SSSS', color='#2196F3', alpha=0.8)
ax1.bar(x + 0.15, speedup_fdm_ffff, 0.3, label='FFFF', color='#E91E63', alpha=0.8)
ax1.set_xlabel('Loading Case')
ax1.set_ylabel('Speedup Factor (×)')
ax1.set_title('FDM-PINN Speedup over Vanilla PINN')
ax1.set_xticks(x)
ax1.set_xticklabels(loads, rotation=20)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
for i in range(5):
    ax1.text(i - 0.15, speedup_fdm[i] + 0.5, f'{speedup_fdm[i]:.0f}×', ha='center', fontsize=8)
    ax1.text(i + 0.15, speedup_fdm_ffff[i] + 0.5, f'{speedup_fdm_ffff[i]:.0f}×', ha='center', fontsize=8)

# Memory reduction
mem_ratio = ssss_gpu[:, 0] / ssss_gpu[:, 1]
mem_ratio_ffff = ffff_gpu[:, 0] / ffff_gpu[:, 1]

ax2.bar(x - 0.15, mem_ratio, 0.3, label='SSSS', color='#2196F3', alpha=0.8)
ax2.bar(x + 0.15, mem_ratio_ffff, 0.3, label='FFFF', color='#E91E63', alpha=0.8)
ax2.set_xlabel('Loading Case')
ax2.set_ylabel('Memory Reduction Factor (×)')
ax2.set_title('GPU Memory Reduction: FDM vs Vanilla')
ax2.set_xticks(x)
ax2.set_xticklabels(loads, rotation=20)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for i in range(5):
    ax2.text(i - 0.15, mem_ratio[i] + 0.3, f'{mem_ratio[i]:.0f}×', ha='center', fontsize=8)
    ax2.text(i + 0.15, mem_ratio_ffff[i] + 0.3, f'{mem_ratio_ffff[i]:.0f}×', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'Speedup_and_memory_reduction.png'))
plt.close()
print("9/10 done")

# ═══════════════════════════════════════════════════════════════
# FIGURE 10: Moment Error Heatmap (FFFF) — Mx RMS
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(np.log10(ffff_mx_rms), cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(4))
ax.set_xticklabels(['Vanilla', 'FDM', 'Fourier', 'Fourier-FDM'], fontsize=10)
ax.set_yticks(range(5))
ax.set_yticklabels(loads, fontsize=10)
for i in range(5):
    for j in range(4):
        ax.text(j, i, f'{ffff_mx_rms[i,j]:.3f}', ha='center', va='center', fontsize=9,
                color='white' if np.log10(ffff_mx_rms[i,j]) > 0 else 'black')
ax.set_title('$M_x$ RMS Error (N·m) — Fully Clamped (FFFF)')
cbar = plt.colorbar(im, ax=ax, label='$\\log_{10}(M_x$ RMS Error$)$')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'FFFF_Mx_RMS_heatmap.png'))
plt.close()
print("10/10 done")

print(f"\nAll 10 comparative figures saved to:\n{OUT}")
print(f"Files: {len(os.listdir(OUT))}")
