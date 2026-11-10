"""Phase 6 -- figures 1, 2, 3 and 5. Figure 4 comes from run_interpretability.py.

    python src/make_figures.py

Every figure is generated from results/tables/*.csv. None is hand-edited.

    fig1_splitting_schematic.png   recording-level vs subject-level splitting
    fig2_accuracy_by_regime.png    accuracy x model x regime, both datasets,
                                   with the majority-class baseline drawn in
    fig3_recording_type.png        subject-level accuracy by recording type
    fig5_hc_recall.png             healthy-control recall, leaky vs honest
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from models import MODEL_ORDER

plt.rcParams.update({
    "figure.dpi": 200, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

TRAIN = "#4c72b0"
TEST = "#dd8452"
LEAKY = "#c44e52"
HONEST = "#4c72b0"
LOSO = "#55a868"
TYPE_ORDER = ["vowel", "number", "sentence", "word"]


def fig1_schematic() -> None:
    """Why the split matters, drawn rather than described."""
    rng = np.random.default_rng(C.SEED)
    n_subj, n_rec = 6, 6
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    for ax, mode in zip(axes, ["recording", "subject"]):
        if mode == "recording":
            # Every recording assigned independently: a subject's recordings
            # land on both sides of the split.
            assign = rng.random((n_subj, n_rec)) < 0.3
            title = "Recording-level split"
            note = "the same speaker appears in train and test"
        else:
            held = np.zeros((n_subj, n_rec), dtype=bool)
            held[rng.choice(n_subj, 2, replace=False), :] = True
            assign = held
            title = "Subject-level split"
            note = "every recording of a speaker stays on one side"

        for i in range(n_subj):
            for j in range(n_rec):
                ax.add_patch(Rectangle(
                    (j, n_subj - i - 1), 0.86, 0.86,
                    facecolor=TEST if assign[i, j] else TRAIN,
                    edgecolor="white", linewidth=1.2))
            ax.text(-0.35, n_subj - i - 1 + 0.43, f"S{i + 1}",
                    ha="right", va="center", fontsize=8)
            if mode == "recording" and assign[i].any() and not assign[i].all():
                ax.text(n_rec + 0.15, n_subj - i - 1 + 0.43, "leak",
                        ha="left", va="center", fontsize=7,
                        color=LEAKY, fontweight="bold")

        ax.set_xlim(-1.2, n_rec + 1.4)
        ax.set_ylim(-0.9, n_subj + 0.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10, pad=8)
        ax.text(n_rec / 2, -0.75, note, ha="center", fontsize=8, style="italic")

    fig.legend(handles=[Patch(facecolor=TRAIN, label="training fold"),
                        Patch(facecolor=TEST, label="test fold")],
               loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Rows are speakers, cells are that speaker's recordings",
                 fontsize=9, y=1.02, color="#444444")
    fig.savefig(C.FIGURES / "fig1_splitting_schematic.png")
    plt.close(fig)


def fig2_accuracy() -> None:
    """The headline contrast, with fold-construction uncertainty shown."""
    summ = pd.read_csv(C.TABLES / "seed_sensitivity_summary.csv")
    cv = pd.read_csv(C.TABLES / "cv_comparison.csv")
    loso = cv[cv.regime == "C_loso"].set_index(["dataset", "model"])["accuracy"]
    # One baseline per dataset; cv has a row per regime, so collapse first.
    base = cv.groupby("dataset")["majority_baseline"].first()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ds in zip(axes, ["UCI/MDVP", "Sakar-2013"]):
        d = summ[summ.dataset == ds].set_index("model").reindex(MODEL_ORDER)
        x = np.arange(len(MODEL_ORDER))
        w = 0.27
        ax.bar(x - w, d.A_mean, w, yerr=d.A_sd, capsize=2.5, color=LEAKY,
               label="A  recording-level (leaky)",
               error_kw={"elinewidth": 0.8, "capthick": 0.8})
        ax.bar(x, d.B_mean, w, yerr=d.B_sd, capsize=2.5, color=HONEST,
               label="B  subject-level 5-fold",
               error_kw={"elinewidth": 0.8, "capthick": 0.8})
        ax.bar(x + w, [loso[(ds, m)] for m in MODEL_ORDER], w, color=LOSO,
               label="C  leave-one-subject-out")

        b = float(base[ds])
        ax.axhline(b, color="#333333", linestyle="--", linewidth=1)
        ax.text(-0.48, b + 0.008, f"majority baseline {b:.3f}",
                ha="left", fontsize=7.5, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in MODEL_ORDER], fontsize=7.5)
        ax.set_ylim(0.40, 1.0)
        ax.set_ylabel("accuracy" if ds == "UCI/MDVP" else "")
        n_rec = cv[cv.dataset == ds].n_recordings.iloc[0]
        n_subj = cv[cv.dataset == ds].n_subjects.iloc[0]
        ax.set_title(f"{ds}  ({n_rec} recordings, {n_subj} subjects)", fontsize=10)

    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left",
                   bbox_to_anchor=(0.0, 0.01))
    fig.suptitle("Accuracy collapses when the split respects the speaker",
                 fontsize=11, y=1.01)
    fig.text(0.5, -0.05,
             "Bars for A and B are means over 10 fold seeds; error bars are 1 SD. "
             "C is deterministic.", ha="center", fontsize=7.5, color="#444444")
    fig.savefig(C.FIGURES / "fig2_accuracy_by_regime.png")
    plt.close(fig)


def fig3_recording_type() -> None:
    """No recording type rescues performance."""
    rt = pd.read_csv(C.TABLES / "recording_type.csv")
    b = rt[rt.regime == "B_group_kfold"]
    mean = b.pivot_table(index="model", columns="rec_type",
                         values="accuracy_mean")[TYPE_ORDER].reindex(MODEL_ORDER)
    sd = b.pivot_table(index="model", columns="rec_type",
                       values="accuracy_sd")[TYPE_ORDER].reindex(MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(MODEL_ORDER))
    w = 0.2
    shades = ["#4c72b0", "#6b93c9", "#93b4dc", "#bcd0ea"]
    for k, (t, c) in enumerate(zip(TYPE_ORDER, shades)):
        ax.bar(x + (k - 1.5) * w, mean[t], w, yerr=sd[t], capsize=2,
               color=c, label=t, error_kw={"elinewidth": 0.7, "capthick": 0.7})

    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    ax.text(len(x) - 0.4, 0.505, "chance 0.500", ha="right", fontsize=7.5,
            color="#333333")
    best = mean.max().max()
    ax.axhline(best, color=LEAKY, linestyle=":", linewidth=1)
    ax.text(-0.45, best + 0.003, f"best cell {best:.3f}", fontsize=7.5, color=LEAKY,
            va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in MODEL_ORDER], fontsize=7.5)
    ax.set_ylim(0.40, 0.68)
    ax.set_ylabel("subject-level accuracy")
    ax.set_title("Sakar: no recording type rescues subject-level performance",
                 fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.04,
             "Regime B, fixed hyper-parameters, means over 10 fold seeds; "
             "error bars are 1 SD.", ha="center", fontsize=7.5, color="#444444")
    fig.savefig(C.FIGURES / "fig3_recording_type.png")
    plt.close(fig)


def fig5_hc_recall() -> None:
    """The specificity collapse that accuracy alone hides."""
    summ = pd.read_csv(C.TABLES / "seed_sensitivity_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, ds in zip(axes, ["UCI/MDVP", "Sakar-2013"]):
        d = summ[summ.dataset == ds].set_index("model").reindex(MODEL_ORDER)
        x = np.arange(len(MODEL_ORDER))
        for i, m in enumerate(MODEL_ORDER):
            a, b = d.recall_hc_A_mean[m], d.recall_hc_B_mean[m]
            ax.annotate("", xy=(i, b), xytext=(i, a),
                        arrowprops={"arrowstyle": "-|>", "color": "#999999",
                                    "linewidth": 1.1})
        ax.scatter(x, d.recall_hc_A_mean, s=46, color=LEAKY, zorder=3,
                   label="recording-level (leaky)")
        ax.scatter(x, d.recall_hc_B_mean, s=46, color=HONEST, zorder=3,
                   label="subject-level")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in MODEL_ORDER], fontsize=7.5)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("healthy-control recall" if ds == "UCI/MDVP" else "")
        ax.set_title(ds, fontsize=10)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle("Controls are what the model stops recognising", fontsize=11,
                 y=1.02)
    fig.savefig(C.FIGURES / "fig5_hc_recall.png")
    plt.close(fig)


def main() -> None:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig1_schematic();      print("fig1_splitting_schematic.png")
    fig2_accuracy();       print("fig2_accuracy_by_regime.png")
    fig3_recording_type(); print("fig3_recording_type.png")
    fig5_hc_recall();      print("fig5_hc_recall.png")
    print(f"wrote to {C.FIGURES.relative_to(C.ROOT)}/")


if __name__ == "__main__":
    main()
