import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
A = "docs/learn/assets"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1) overall gain by method (verified from reports/eval_*; unet_hq from checkpoint val)
methods = ["B1 best single tool", "Agent (RL)", "Agent vanilla", "Oracle +STOP\n(tool-chain ceiling)",
           "Agent fine-tuned", "Monolith (1 CNN)"]
gains = [2.34, 2.93, 2.94, 3.24, 3.67, 4.03]
colors = ["#9aa4b8", "#4f7cff", "#7aa0ff", "#b0b0b0", "#9b6bff", "#3dd68c"]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.barh(range(len(methods)), gains, color=colors)
ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods); ax.invert_yaxis()
ax.axvline(3.24, ls="--", c="#999", lw=1)
for i, g in enumerate(gains):
    ax.text(g + 0.04, i, f"+{g:.2f}", va="center", fontsize=9)
ax.set_xlabel("PSNR gain over degraded input (dB) — DIV2K 751–800 held-out test")
ax.set_title("How much each method improves a damaged photo (higher = better)")
ax.set_xlim(0, 4.7)
ax.text(3.24, -0.5, "tool-chain ceiling (oracle +3.24)", ha="center", fontsize=8, color="#777")
plt.tight_layout(); plt.savefig(f"{A}/results_overall.png", dpi=130); plt.close()

# 2) per-severity (only reports/-verified methods)
sev = ["mild", "moderate", "severe"]
data = {"Agent (RL)": [2.24, 2.83, 3.71], "Agent fine-tuned": [2.88, 3.57, 4.55],
        "Monolith": [3.38, 3.91, 4.79], "Oracle +STOP": [2.73, 3.13, 3.86]}
cset = ["#4f7cff", "#9b6bff", "#3dd68c", "#b0b0b0"]
x = np.arange(3); w = 0.2
fig, ax = plt.subplots(figsize=(9, 4.4))
for i, (k, v) in enumerate(data.items()):
    ax.bar(x + (i - 1.5) * w, v, w, label=k, color=cset[i])
ax.set_xticks(x); ax.set_xticklabels(sev); ax.set_ylabel("PSNR gain (dB)")
ax.set_title("Gain by damage severity — every method helps more on worse damage")
ax.legend(fontsize=9, ncol=2)
plt.tight_layout(); plt.savefig(f"{A}/results_by_severity.png", dpi=130); plt.close()

# 3) forward selection — toolbox size
k = [1, 2, 3, 4, 5, 6, 7, 8, 9]
pct = [65.6, 84.5, 91.9, 95.9, 97.9, 98.8, 99.4, 99.6, 99.9]
fig, ax = plt.subplots(figsize=(8, 4.3))
ax.plot(k, pct, "-o", color="#4f7cff")
ax.axhline(95.9, ls=":", c="#1d9e75"); ax.axvline(4, ls=":", c="#1d9e75")
ax.annotate("4 tools reach 95.9% of the\nfull 12-tool ceiling", (4, 95.9), xytext=(4.7, 86),
            fontsize=9, color="#1d9e75", arrowprops=dict(arrowstyle="->", color="#1d9e75"))
ax.set_xlabel("number of tools (best k, chosen greedily)")
ax.set_ylabel("% of the full 12-tool ceiling")
ax.set_title("Most of the gain comes from ~4–5 tools (forward selection)")
ax.set_ylim(60, 102)
plt.tight_layout(); plt.savefig(f"{A}/toolbox_forward_selection.png", dpi=130); plt.close()
print("generated 3 plots in", A)
