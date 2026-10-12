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
    question(
        ax,
        50,
        89,
        "Q1 · Does it make a concrete claim about this conversation or a subject it raises?",
    )
    question(ax, 50, 78.5, "Q2 · Is it a list of options?  (“Germany, France or UK”)")
    question(
        ax,
        50,
        68,
        "Q3 · Is it about the model’s own beliefs or intent, or framing "
        "the chat as a test / roleplay?",
    )
    question(ax, 50, 57.5, "Q4 · Can you quote the conversation to settle it?")
    question(ax, 20, 42.5, "Q5 · What is wrong?", w=30, width=24)
    question(ax, 83, 45, "Q6 · Is it a world fact or about the conversation?", w=30, width=26)

    # side leaves
    card(
        ax,
        9,
        80.5,
        "JUNK",
        "junk",
        "Asserts nothing, or is about something unrelated. "
        "Tag: off_topic, generic, filler, formatting, junk text, lens meta.",
        "“a friendly reply” · bullets · “the tokens include ‘холод’”",
        h=23,
    )
    card(
        ax,
        91,
        89,
        "PROBE CANDIDATE",
        "ambiguous_topic",
        "Can’t tell whether it refers to the conversation.",
        "",
        h=12,
    )
    card(
        ax,
        91,
        72.3,
        "UNRESOLVED",
        "unverifiable",
        "Option lists, whether or not the right one is listed (tag: same-slot alternative).",
        "“pepperoni, veggie or Hawaiian?”",
        h=19,
    )
    card(
        ax,
        9,
        57,
        "THINKY",
        "model_state",
        "The model’s own beliefs, intent, awareness; test / evaluation / roleplay framing.",
        "“I will assume this is a conceptual database exercise”",
        h=19,
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
        "“Miss Wang” for Miss Li · “float voltage” for double",
    )
    card(
        ax,
        18.8,
        y,
        "HALLUCINATION",
        "false_claim",
        "Wrong proposition or situation: reversed outcome, wrong agent, different answer, "
        "wrong scenario.",
        "“flood or sewage backup” in a tobacco-smell chat",
    )
    card(
        ax,
        31.1,
        y,
        "HALLUCINATION",
        "substitution_category",
        "A different object of a different category in place of the right one.",
        "“video game” for movie · Nazis for the Mafia",
    )
    card(
        ax,
        43.4,
        y,
        "PROBE CANDIDATE",
        "substitution_sibling",
        "A different object of the same class; may be real co-activation.",
        "“Germany” for France · “I’m ChatGPT” (it is Qwen)",
    )
    card(
        ax,
        55.7,
        y,
        "CORRECT",
        "correct",
        "Stated before the marker, realised after it in any wording, or strictly entailed. "
        "Quote required.",
        "“the engineers who created me”",
    )
    card(
        ax,
        68,
        y,
        "UNRESOLVED",
        "unverifiable",
        "About the conversation; not contradicted and not settled.",
        "“the spill has been cleaned up” · “welcome to Oakville!”",
    )
    card(
        ax,
        80.3,
        y,
        "CORRECT",
        "supplementary_true",
        "World fact the text never mentions, and it’s true.",
        "true fact about Picasso the text skips",
    )
    card(
        ax,
        92.6,
        y,
        "PROBE CANDIDATE",
        "supplementary_false",
        "World fact the text never mentions, and it’s false: possible model misconception.",
        "“water flushes out toxins”",
    )

    # arrows: Q1
    arrow(ax, 33, 89, 15, 87.5, "No / unrelated", 24, 89.6)
    arrow(ax, 67, 89, 85, 89, "Can’t tell", 76, 90.5)
    arrow(ax, 50, 85.8, 50, 81.7, "Yes", 52.2, 83.8)
    # Q2
    arrow(ax, 67, 78.5, 85, 77.5, "Yes", 76, 79.5)
    arrow(ax, 50, 75.3, 50, 71.2, "No", 52, 73.2)
    # Q3
    arrow(ax, 33, 68, 15, 62, "Yes", 24, 66.5)
    arrow(ax, 50, 64.8, 50, 60.7, "No", 52, 62.7)
    # Q4 branches
    arrow(ax, 38, 55, 22, 45.9, "Obviously contradicted\n(quote required)", 27, 53)
    arrow(ax, 50, 54.3, 55.7, 31.5, "Stated / realised /\nentailed (quote)", 55, 44)
    arrow(ax, 62, 55, 80, 48.2, "Text is silent", 73, 53.2)
    # Q5
    for x in (6.5, 18.8, 31.1, 43.4):
        arrow(ax, 20, 39.3, x, 31.5)
    ax.text(10.5, 35.6, "wrong\nspecific", fontsize=9, color=MUTED, ha="center")
    ax.text(17.2, 34.6, "wrong\nclaim", fontsize=9, color=MUTED, ha="center")
    ax.text(27.5, 35.6, "other\ncategory", fontsize=9, color=MUTED, ha="center")
    ax.text(36.5, 37.2, "same-class\nobject", fontsize=9, color=MUTED, ha="center")
    # Q6
    arrow(ax, 76, 41.8, 68, 31.5, "about the\nconversation", 70.5, 37.2)
    arrow(ax, 83, 41.8, 80.3, 31.5, "true", 81.2, 36.5)
    arrow(ax, 90, 41.8, 92.6, 31.5, "false", 94.6, 37.2)

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
