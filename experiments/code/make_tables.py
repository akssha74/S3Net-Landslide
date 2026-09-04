#!/usr/bin/env python3
"""Generate LaTeX tables directly from verified experimental results on HR-GLDD."""

import os
import json

RESULTS_DIR = "studies/disaster-hrgldd-landslide/experiments/derived/results"
TABLES_DIR = "studies/disaster-hrgldd-landslide/paper/tables"
os.makedirs(TABLES_DIR, exist_ok=True)

def generate_performance_table():
    summary_file = os.path.join(RESULTS_DIR, "rigorous_confirmatory_summary.json")
    with open(summary_file) as f:
        data = json.load(f)["summary_by_arm"]
        
    tex = r"""\begin{tabular}{lcccccc}
\toprule
\textbf{Architecture / Model} & \textbf{Params} & \textbf{Micro F1 (\%)} & \textbf{Macro F1 (\%)} & \textbf{IoU (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} \\
\midrule
"""
    arm_names = [
        ("unet", "Vanilla U-Net~\\cite{meena2023hrgldd,ronneberger2015unet}", "1.93M"),
        ("resunet", "ResU-Net Baseline~\\cite{meena2023hrgldd,zhang2018resunet}", "2.01M"),
        ("s3net_ablation", "S$^3$-Net (w/o Physical NDVI Gating)", "2.11M"),
        ("s3net", "\\textbf{S$^3$-Net (Proposed)}", "\\textbf{2.11M}")
    ]
    
    for key, display, p_count in arm_names:
        row = data[key]
        f1_val = row["micro_f1_mean"] * 100
        f1_std = row["micro_f1_std"] * 100
        macro_val = row["macro_f1_mean"] * 100
        macro_std = row["macro_f1_std"] * 100
        iou_val = row["micro_iou_mean"] * 100
        iou_std = row["micro_iou_std"] * 100
        prec_val = row["micro_precision_mean"] * 100
        prec_std = row["micro_precision_std"] * 100
        rec_val = row["micro_recall_mean"] * 100
        rec_std = row["micro_recall_std"] * 100
        
        if key == "s3net":
            f1_str = f"$\\mathbf{{{f1_val:.2f} \\pm {f1_std:.2f}}}$"
            macro_str = f"$\\mathbf{{{macro_val:.2f} \\pm {macro_std:.2f}}}$"
            iou_str = f"$\\mathbf{{{iou_val:.2f} \\pm {iou_std:.2f}}}$"
            rec_str = f"$\\mathbf{{{rec_val:.2f} \\pm {rec_std:.2f}}}$"
            prec_str = f"${prec_val:.2f} \\pm {prec_std:.2f}$"
        elif key == "s3net_ablation":
            f1_str = f"${f1_val:.2f} \\pm {f1_std:.2f}$"
            macro_str = f"${macro_val:.2f} \\pm {macro_std:.2f}$"
            iou_str = f"${iou_val:.2f} \\pm {iou_std:.2f}$"
            rec_str = f"${rec_val:.2f} \\pm {rec_std:.2f}$"
            prec_str = f"$\\mathbf{{{prec_val:.2f} \\pm {prec_std:.2f}}}$"
        else:
            f1_str = f"${f1_val:.2f} \\pm {f1_std:.2f}$"
            macro_str = f"${macro_val:.2f} \\pm {macro_std:.2f}$"
            iou_str = f"${iou_val:.2f} \\pm {iou_std:.2f}$"
            prec_str = f"${prec_val:.2f} \\pm {prec_std:.2f}$"
            rec_str = f"${rec_val:.2f} \\pm {rec_std:.2f}$"
            
        tex += f"{display} & {p_count} & {f1_str} & {macro_str} & {iou_str} & {prec_str} & {rec_str} \\\\\n"
        
    tex += r"""\midrule
DCA-UNet (Song et al., 2026~\cite{dcaunet2026}) & 29.50M & 74.41 & -- & 59.24 & -- & -- \\
\bottomrule
\end{tabular}
"""
    out_file = os.path.join(TABLES_DIR, "tab_performance.tex")
    with open(out_file, "w") as f:
        f.write(tex)
    print(f"Generated {out_file}")

if __name__ == "__main__":
    generate_performance_table()
