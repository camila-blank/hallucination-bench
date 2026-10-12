"""Infographic of the claim-map judge tree (plan 0002 rev 20).

uv run python scripts/plot_judge_tree.py [--out plots/judge_tree.png]
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
TEXT, MUTED = "#0b0b0b", "#52514e"
VERDICT = {
    "HALLUCINATION": "#e34948",
    "PROBE CANDIDATE": "#4a3aa7",
    "THINKY": "#d55181",
    "CORRECT": "#1baf7a",
    "UNRESOLVED": "#2a78d6",
    "JUNK": "#8c8b85",
}


def wrap(s: str, w: int) -> str:
    return "\n".join(textwrap.wrap(s, w))


def question(ax, x, y, text, w=44, width=34):
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - 3.2),
            width,
            6.4,
            boxstyle="round,pad=0.25,rounding_size=1.2",
            fc="#f4f3ef",
            ec=MUTED,
            lw=1.2,
        )
    )
    ax.text(x, y, wrap(text, w), ha="center", va="center", fontsize=11.5, color=TEXT)


def card(ax, x, y, verdict, leaf, definition, example, w=11.6, h=19, wrapw=24):
    c = VERDICT[verdict]
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.2,rounding_size=1.0",
            fc="white",
            ec=c,
            lw=2.2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y + h / 2 - 3.4),
            w,
            3.4,
            boxstyle="round,pad=0.2,rounding_size=1.0",
            fc=c,
            ec=c,
            lw=2.2,
        )
    )
    ax.text(
        x,
        y + h / 2 - 1.7,
        verdict,
        ha="center",
        va="center",
        fontsize=8.5,
        color="white",
        fontweight="bold",
    )
    ax.text(
        x,
        y + h / 2 - 5.4,
        leaf,
        ha="center",
        va="center",
        fontsize=11.5 if len(leaf) <= 14 else 9.2,
        color=TEXT,
        fontweight="bold",
        family="monospace",
    )
    ax.text(
        x,
        y + h / 2 - 7.4,
        wrap(definition, wrapw),
        ha="center",
        va="top",
        fontsize=8.8,
        color=TEXT,
        linespacing=1.25,
    )
    ax.text(
        x,
        y - h / 2 + 1.0,
        wrap(example, wrapw + 2),
        ha="center",
        va="bottom",
        fontsize=8.3,
        color=MUTED,
        style="italic",
        linespacing=1.2,
    )


def arrow(ax, x0, y0, x1, y1, label="", lx=None, ly=None, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            lw=1.4,
            color=MUTED,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        ax.text(
            lx if lx is not None else (x0 + x1) / 2,
            ly if ly is not None else (y0 + y1) / 2,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color=TEXT,
            bbox={"fc": "white", "ec": "none", "pad": 1.5},
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "plots" / "judge_tree.png"))
    out = Path(ap.parse_args().out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 14), dpi=170)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(
        1,
        98.5,
        "HallucinationBench v2 · how the judge labels every piece of a lens readout",
        fontsize=19,
        color=TEXT,
        fontweight="bold",
        va="top",
    )
    ax.text(
        1,
        95.6,
        "The readout is split into sentence/bullet units; the judge gives each unit (or a "
        "quoted sub-span) exactly one leaf. Each leaf maps to one verdict.",
        fontsize=11.5,
        color=MUTED,
        va="top",
    )

    # questions
    question(ax, 50, 88, "Q1 · Is it a claim?", w=40, width=26)
    question(ax, 50, 74, "Q2 · Is it about the model\u2019s state?", w=40, width=30)
    question(ax, 50, 60, "Q3 · Is it verifiable from the text?", w=40, width=30)
    question(ax, 30, 45, "Q4 · Is it correct?", w=30, width=22)

    # side leaves
    card(
        ax,
        12,
        82,
        "JUNK",
        "junk",
        "Not a claim, or about something unrelated.",
        "\u201ca friendly reply\u201d · bullets · byte garbage",
        h=15,
    )
    card(
        ax,
        88,
        88,
        "PROBE CANDIDATE",
        "ambiguous_topic",
        "Can\u2019t tell whether it refers to the conversation.",
        "",
        h=12,
    )
    card(
        ax,
        12,
        64,
        "THINKY",
        "model_state",
        "The model\u2019s own beliefs, intent or awareness; test / roleplay framing.",
        "\u201cI will assume this is a conceptual database exercise\u201d",
        h=17,
    )

    # bottom row
    y = 22
    card(
        ax,
        6.5,
        y,
        "HALLUCINATION",
        "detail",
        "Right reading, wrong specific: name, number, date, wording, context word.",
        "\u201cMiss Wang\u201d for Miss Li · \u201cfloat voltage\u201d for double",
    )
    card(
        ax,
        18.8,
        y,
        "HALLUCINATION",
        "false_claim",
        "Wrong proposition or situation: reversed outcome, wrong agent, different answer, "
        "wrong scenario.",
        "\u201cflood or sewage backup\u201d in a tobacco-smell chat",
    )
    card(
        ax,
        31.1,
        y,
        "HALLUCINATION",
        "substitution_category",
        "A different object of a different category in place of the right one.",
        "\u201cvideo game\u201d for movie · Nazis for the Mafia",
    )
    card(
        ax,
        43.4,
        y,
        "PROBE CANDIDATE",
        "substitution_sibling",
        "A different object of the same class; may be real co-activation.",
        "\u201cGermany\u201d for France · \u201cI\u2019m ChatGPT\u201d (it is Qwen)",
    )
    card(
        ax,
        55.7,
        y,
        "CORRECT",
        "correct",
        "Stated before the marker, realised after it in any wording, or strictly entailed.",
        "\u201cthe engineers who created me\u201d",
    )
    card(
        ax,
        68,
        y,
        "UNRESOLVED",
        "unverifiable",
        "About the conversation, but the text can\u2019t settle it (incl. lists of options).",
        "\u201cthe spill has been cleaned up\u201d · \u201cpepperoni or veggie?\u201d",
    )
    card(
        ax,
        80.3,
        y,
        "CORRECT",
        "supplementary_true",
        "A world fact the text never mentions, and it\u2019s true.",
        "true fact about Picasso the text skips",
    )
    card(
        ax,
        92.6,
        y,
        "PROBE CANDIDATE",
        "supplementary_false",
        "A world fact the text never mentions, and it\u2019s false: possible misconception.",
        "\u201cwater flushes out toxins\u201d",
    )

    # arrows
    arrow(ax, 37, 88, 18, 86, "No", 27, 88.6)
    arrow(ax, 63, 88, 82, 88, "Can\u2019t tell", 72.5, 89.6)
    arrow(ax, 50, 84.8, 50, 77.2, "Yes", 52.2, 81)
    arrow(ax, 35, 74, 18, 67, "Yes", 26, 72)
    arrow(ax, 50, 70.8, 50, 63.2, "No", 52, 67)
    arrow(ax, 40, 57, 31, 48.2, "Yes (quote)", 33, 53.5)
    arrow(ax, 41, 45, 55.7, 31.5, "Yes", 50, 39)
    for x, lab, lx, ly in (
        (6.5, "wrong specific", 10.5, 36.8),
        (18.8, "wrong claim", 19.6, 35.4),
        (31.1, "other category", 30.5, 36.2),
        (43.4, "same-class object", 41.5, 38.2),
    ):
        arrow(ax, 26, 41.8, x, 31.5)
        ax.text(lx, ly, wrap(lab, 10), fontsize=9, color=MUTED, ha="center")
    ax.text(
        23,
        39.6,
        "No",
        fontsize=10,
        color=TEXT,
        ha="center",
        bbox={"fc": "white", "ec": "none", "pad": 1.5},
    )
    arrow(ax, 61, 57, 68, 31.5, "No \u00b7 about the\nconversation", 64.8, 44)
    arrow(ax, 64, 58, 80.3, 31.5, "No \u00b7 world fact,\ntrue", 72.5, 44)
    arrow(ax, 65, 59, 92.6, 31.5, "No \u00b7 world fact,\nfalse", 82.5, 42)

    # legend
    x = 1
    ax.text(x, 6.6, "Verdicts:", fontsize=11.5, color=TEXT, fontweight="bold", va="center")
    x += 7.5
    for v, c in VERDICT.items():
        ax.add_patch(FancyBboxPatch((x, 5.4), 2.0, 2.4, boxstyle="round,pad=0.1", fc=c, ec=c))
        ax.text(
            x + 2.6,
            6.6,
            v.capitalize() if v != "PROBE CANDIDATE" else "Probe candidate",
            fontsize=11,
            color=TEXT,
            va="center",
        )
        x += 15
    ax.text(
        1,
        2.2,
        "Hallucination = obviously contradicted by the text, with the contradicting span "
        "quoted (no valid quote → unverifiable). Continuation-style readouts go through the "
        "same gate as predictions of the response. Judge claimmap-v2.7, scoring rev 20.",
        fontsize=10,
        color=MUTED,
        va="center",
    )
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
