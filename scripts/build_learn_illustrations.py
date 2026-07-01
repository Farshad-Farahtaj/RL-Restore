import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
A = "docs/learn/assets"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

def psnr(a, b):
    m = np.mean((np.clip(a,0,1) - np.clip(b,0,1)) ** 2)
    return 99.0 if m < 1e-9 else 10 * np.log10(1.0 / m)

img = np.asarray(Image.open("data/DIV2K/DIV2K_train_HR/0763.png").convert("RGB"), np.float32) / 255
def gblur(x, s):
    from PIL import ImageFilter
    return np.asarray(Image.fromarray((np.clip(x,0,1)*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(s)), np.float32)/255

def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{A}/{name}", dpi=130); plt.close(fig); print("ok", name)

# 1 L1 vs L2
try:
    x = np.linspace(-2, 2, 200)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(x, np.abs(x), label="L1 = |error|  (MAE)", lw=2, color="#4f7cff")
    ax.plot(x, x**2, label="L2 = error²  (MSE)", lw=2, color="#9b6bff")
    ax.set_xlabel("prediction error"); ax.set_ylabel("loss (penalty)")
    ax.set_title("L1 vs L2 loss — L2 punishes big errors far harder")
    ax.legend(); ax.set_ylim(0, 4)
    save(fig, "loss_l1_l2.png")
except Exception as e: print("FAIL l1l2", e)

# 2 Q-values bar chart (illustrative example of one decision)
try:
    labels = ["dbl1","dbl2","dbl3","dbl4","dn1","dn2","dn3","dn4","jp1","jp2","jp3","jp4","STOP"]
    q = np.array([-0.4,-0.2,0.1,-0.3,0.3,2.1,0.9,0.2,-0.1,0.0,0.1,0.05,0.42])
    cols = ["#cfd6e6"]*13; cols[5] = "#4f7cff"; cols[12] = "#3dd68c"
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(range(13), q, color=cols)
    ax.set_xticks(range(13)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_ylabel("Q-value (expected future dB)")
    ax.set_title("The agent scores all 13 actions, then picks the highest (here: denoise2)")
    ax.annotate("chosen", (5, 2.1), xytext=(7, 1.9), fontsize=9, color="#4f7cff", arrowprops=dict(arrowstyle="->", color="#4f7cff"))
    ax.text(12, 0.55, "STOP", fontsize=8, color="#1d9e75", ha="center")
    save(fig, "q_values.png")
except Exception as e: print("FAIL q", e)

# 3 three-axis radar
try:
    axes = ["Accuracy\n(PSNR)", "Accountability\n(explains itself)", "Perceptual\n(looks crisp)"]
    series = {"RL Agent": [0.55, 1.0, 0.35], "Monolith (1 CNN)": [0.95, 0.1, 0.4],
              "Real-ESRGAN enhance": [0.5, 0.1, 0.95]}
    cset = {"RL Agent": "#4f7cff", "Monolith (1 CNN)": "#3dd68c", "Real-ESRGAN enhance": "#9b6bff"}
    ang = np.linspace(0, 2*np.pi, len(axes), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6.2, 5.2), subplot_kw=dict(polar=True))
    for k, v in series.items():
        vv = v + v[:1]
        ax.plot(ang, vv, label=k, color=cset[k], lw=2); ax.fill(ang, vv, color=cset[k], alpha=0.08)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(axes, fontsize=9); ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.set_title("No single winner: each method leads on a different axis", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), fontsize=8)
    save(fig, "three_axis_radar.png")
except Exception as e: print("FAIL radar", e)

# 4 white-world bug
try:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    steps = np.linspace(0, 2, 100); loss = 0.9*np.exp(-steps*2.2) + 0.02 + np.random.default_rng(0).normal(0, 0.01, 100)
    a1.plot(steps, loss, color="#3dd68c"); a1.set_title("What the metrics showed (looks great)")
    a1.set_xlabel("training (millions of steps)"); a1.set_ylabel("training loss"); a1.set_ylim(0, 1)
    a2.imshow(np.ones((63, 63, 3))); a2.set_title("What the model actually output")
    a2.text(31, 33, "blank white", ha="center", color="#888"); a2.set_xticks([]); a2.set_yticks([])
    fig.suptitle("The uint8 'white-world' bug — a green report card on a blank page", fontsize=11)
    save(fig, "white_world_bug.png")
except Exception as e: print("FAIL white", e)

# 5 RGB channels
try:
    c = img[200:520, 200:520]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.1))
    axs[0].imshow(c); axs[0].set_title("the colour image")
    for i, (nm, cmap) in enumerate([("Red", "Reds"), ("Green", "Greens"), ("Blue", "Blues")]):
        axs[i+1].imshow(c[:, :, i], cmap=cmap, vmin=0, vmax=1); axs[i+1].set_title(f"{nm} channel\n(a grid of numbers 0–1)")
    for a in axs: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("A photo is three stacked grids of numbers (one per colour)", fontsize=11)
    save(fig, "rgb_channels.png")
except Exception as e: print("FAIL rgb", e)

# 6 image as grid of numbers
try:
    g = (img[300:308, 300:308, 0])  # 8x8, red channel
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.4))
    a1.imshow(np.kron(g, np.ones((40, 40))), cmap="gray", vmin=0, vmax=1)
    a1.set_title("an 8×8 patch (zoomed in)"); a1.set_xticks([]); a1.set_yticks([])
    a2.imshow(np.ones_like(g), cmap="gray", vmin=0, vmax=1)
    for i in range(8):
        for j in range(8):
            a2.text(j, i, f"{int(g[i,j]*255)}", ha="center", va="center", fontsize=8)
    a2.set_title("...is just these numbers (0–255)"); a2.set_xticks([]); a2.set_yticks([])
    a2.set_xticks(np.arange(-.5, 8, 1), minor=True); a2.set_yticks(np.arange(-.5, 8, 1), minor=True)
    a2.grid(which="minor", color="#ccc", lw=0.5)
    fig.suptitle("To a computer, an image is literally a grid of numbers", fontsize=11)
    save(fig, "image_as_grid.png")
except Exception as e: print("FAIL grid", e)

# 7 convolution schematic
try:
    fig, ax = plt.subplots(figsize=(8.5, 4.2)); ax.axis("off")
    rng = np.random.default_rng(1); inp = rng.integers(0, 9, (5, 5))
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((j, 4-i), 1, 1, fill=False, ec="#bbb"))
            ax.text(j+.5, 4-i+.5, inp[i, j], ha="center", va="center", fontsize=9, color="#666")
    for (i, j) in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]:
        ax.add_patch(Rectangle((j, 4-i), 1, 1, fill=True, fc="#4f7cff22", ec="#4f7cff", lw=2))
    ax.annotate("", xy=(7.2, 2.5), xytext=(5.4, 2.5), arrowprops=dict(arrowstyle="->", lw=2, color="#666"))
    ax.text(6.3, 2.9, "3×3 kernel\n(weights)", ha="center", fontsize=9, color="#4f7cff")
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((8+j, 3-i), 1, 1, fill=False, ec="#bbb"))
    ax.add_patch(Rectangle((8, 3), 1, 1, fill=True, fc="#3dd68c33", ec="#3dd68c", lw=2))
    ax.text(9.5, 4.4, "output (feature map)", ha="center", fontsize=9, color="#1d9e75")
    ax.text(2.5, -0.6, "the same kernel slides over every position →", ha="center", fontsize=9, color="#666")
    ax.set_xlim(-0.3, 11.3); ax.set_ylim(-1, 5.2)
    ax.set_title("A convolution: one small kernel slides across the image")
    save(fig, "convolution.png")
except Exception as e: print("FAIL conv", e)

# 8 regression to the mean
try:
    base = img[260:340, 360:440]
    cands = [np.roll(np.roll(base, dx, 0), dy, 1) for dx, dy in [(-3,-3),(3,-3),(-3,3),(3,3),(0,0)]]
    avg = np.mean(cands, axis=0)
    fig = plt.figure(figsize=(10, 4.2))
    for i, c in enumerate(cands[:4]):
        ax = fig.add_subplot(2, 4, i+1); ax.imshow(np.clip(c,0,1)); ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_ylabel("plausible\noriginals", fontsize=9)
        ax.set_title("candidate", fontsize=8)
    axavg = fig.add_subplot(1, 4, 4); axavg.imshow(np.clip(avg,0,1)); axavg.set_xticks([]); axavg.set_yticks([])
    axavg.set_title("their average\n= the 'safe' guess\n(blurry!)", fontsize=9, color="#c0392b")
    fig.suptitle("Regression to the mean: averaging many sharp possibilities gives a blurry result", fontsize=11)
    save(fig, "regression_to_mean.png")
except Exception as e: print("FAIL reg2mean", e)

# 9 PSNR blind spot
try:
    from PIL import ImageFilter
    clean = img[300:460, 300:460]
    soft = gblur(clean, 0.9)
    pil = Image.fromarray((np.clip(clean,0,1)*255).astype(np.uint8))
    crisp = np.asarray(pil.filter(ImageFilter.UnsharpMask(radius=2, percent=220)), np.float32)/255
    ps, pc = psnr(soft, clean), psnr(crisp, clean)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.8))
    axs[0].imshow(clean); axs[0].set_title("clean reference")
    axs[1].imshow(np.clip(soft,0,1)); axs[1].set_title(f"soft estimate\nPSNR {ps:.1f} dB (higher!)", color="#1d9e75")
    axs[2].imshow(np.clip(crisp,0,1)); axs[2].set_title(f"crisp estimate\nPSNR {pc:.1f} dB (lower)", color="#c0392b")
    for a in axs: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("PSNR's blind spot: the softer image scores higher, yet looks worse", fontsize=11)
    save(fig, "psnr_blindspot.png")
except Exception as e: print("FAIL psnr", e)
print("done")
