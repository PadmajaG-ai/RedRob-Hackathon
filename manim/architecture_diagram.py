"""
Manim animation of the AI Candidate Ranking end-to-end architecture.

Based on generate_submission.py, llm_judge.py, train_ltr_model.py, and fairness_audit.py.

Render:
    manim -qh  manim/architecture_diagram.py ArchitectureDiagram         # MP4
    manim -qh -s manim/architecture_diagram.py ArchitectureDiagramStatic # PNG
"""

from manim import *


# ── Palette ─────────────────────────────────────────────────────────────────
BG       = "#0f1117"
C_INPUT  = "#3b82f6"
C_STAGE1 = "#22c55e"
C_STAGE2 = "#a855f7"
C_STAGE3 = "#2563eb"
C_OUTPUT = "#ef4444"
C_AUDIT  = "#d97706"
C_TEXT   = WHITE
C_MUTED  = "#94a3b8"
C_ARROW  = "#64748b"


# ── Helper builders ──────────────────────────────────────────────────────────

def component_box(title, subtitle, tag, accent,
                  width=3.4, height=1.35, tag_color=None):
    tag_color = tag_color or accent
    rect = RoundedRectangle(
        corner_radius=0.12, width=width, height=height,
        fill_color=accent, fill_opacity=0.18,
        stroke_color=accent, stroke_width=2,
    )
    title_m = Text(title, font_size=17, color=C_TEXT, weight=BOLD).move_to(
        rect.get_center() + UP * 0.22)
    sub_m = Text(subtitle, font_size=11, color=C_MUTED, line_spacing=0.9).move_to(
        rect.get_center() + DOWN * 0.28)
    tag_bg = RoundedRectangle(
        corner_radius=0.06, width=0.95, height=0.28,
        fill_color=tag_color, fill_opacity=1, stroke_width=0,
    ).move_to(rect.get_corner(UL) + RIGHT * 0.55 + DOWN * 0.18)
    tag_m = Text(tag, font_size=10, color=WHITE, weight=BOLD).move_to(tag_bg.get_center())
    return VGroup(rect, title_m, sub_m, tag_bg, tag_m)


def stage_banner(label, color, width=9.8):
    bar = RoundedRectangle(
        corner_radius=0.08, width=width, height=0.40,
        fill_color=color, fill_opacity=0.32,
        stroke_color=color, stroke_width=1.5,
    )
    txt = Text(label, font_size=13, color=color, weight=BOLD).move_to(bar.get_center())
    return VGroup(bar, txt)


def audit_box(title, subtitle):
    inner = RoundedRectangle(
        corner_radius=0.12, width=2.45, height=1.15,
        fill_color=C_AUDIT, fill_opacity=0.12,
        stroke_color=C_AUDIT, stroke_width=2,
    )
    rect = DashedVMobject(inner, num_dashes=18)
    t = Text(title, font_size=13, color=C_AUDIT, weight=BOLD).move_to(
        inner.get_center() + UP * 0.17)
    s = Text(subtitle, font_size=10, color=C_MUTED, line_spacing=0.85).move_to(
        inner.get_center() + DOWN * 0.2)
    return VGroup(rect, t, s)


def flow_label(text, pos):
    return Text(text, font_size=10, color=C_MUTED).move_to(pos)


# ── Layout constants (unscaled Manim units, top → bottom) ───────────────────
#
#  y =  3.3  Title
#  y =  2.5  [JD Input | Candidates]          ← Row 0 (inputs)
#  y =  1.4  [JD Parser]   (left col only)    ← Row 1
#  y =  0.65 Stage 1 banner
#  y = -0.15 [Dense | BM25]                   ← Row 2
#  y = -1.45 [RRF Fusion]                     ← Row 3
#  y = -2.3  Stage 2 banner
#  y = -3.2  [CatBoost | 6-Dim Scoring]       ← Row 4
#  y = -4.1  Stage 3 banner
#  y = -5.0  [Qwen2.5-3B | Template]          ← Row 5
#  y = -6.2  [submission_v4.csv]              ← Row 6 (output)
#
#  Pool Audit:   x = -6.1, y = -0.8
#  Output Audit: x =  6.1, y = -5.0
#  Legend:       bottom-right corner
#
#  Main column x range: [-4.5, 4.5]    (JD input left = -4.5, BM25 right ≈ 4.5)


def build_diagram():

    # ── Title ────────────────────────────────────────────────────────────────
    title = Text(
        "AI Candidate Ranking — End-to-End Architecture",
        font_size=26, color=C_TEXT, weight=BOLD,
    ).move_to(UP * 3.3)

    # ── Row 0: Inputs ────────────────────────────────────────────────────────
    jd_input = component_box(
        "Job Description", "job_description.txt", "INPUT", C_INPUT, width=3.0,
    ).move_to(LEFT * 4.2 + UP * 2.5)

    candidates = component_box(
        "100K Candidates", "eda.load_data()", "INPUT", C_INPUT, width=3.0,
    ).move_to(RIGHT * 0.8 + UP * 2.5)

    # ── Row 1: JD Parser (left side) ─────────────────────────────────────────
    jd_parser = component_box(
        "JD Parser",
        "llm_judge.py  ·  skills · years · seniority",
        "PARSE", C_INPUT, width=3.2,
    ).move_to(LEFT * 4.2 + UP * 1.35)

    # ── Stage 1 banner ───────────────────────────────────────────────────────
    s1_banner = stage_banner(
        "STAGE 1 — Hybrid Retrieval", C_STAGE1,
    ).move_to(UP * 0.58)

    # ── Row 2: Retrieval boxes ───────────────────────────────────────────────
    dense = component_box(
        "Dense Retrieval",
        "all-MiniLM-L6-v2 + FAISS\nprecomputed 100K embeddings",
        "SEMANTIC", C_STAGE1, width=3.5, height=1.4,
    ).move_to(LEFT * 2.4 + DOWN * 0.22)

    bm25 = component_box(
        "BM25 Sparse Retrieval",
        "rank_bm25 on 100K candidates\nlexical exact-skill matching",
        "LEXICAL", C_STAGE1, width=3.5, height=1.4,
    ).move_to(RIGHT * 2.4 + DOWN * 0.22)

    # ── Row 3: RRF Fusion ────────────────────────────────────────────────────
    rrf = component_box(
        "RRF Fusion",
        "Reciprocal Rank Fusion  ·  ~1000 candidate pool",
        "MERGE", C_STAGE1, width=4.2, height=1.25,
    ).move_to(DOWN * 1.48)

    # ── Stage 2 banner ───────────────────────────────────────────────────────
    s2_banner = stage_banner(
        "STAGE 2 — CatBoost LTR Re-ranking", C_STAGE2,
    ).move_to(DOWN * 2.42)

    # ── Row 4: LTR ───────────────────────────────────────────────────────────
    catboost = component_box(
        "CatBoostRanker (YetiRank)",
        "ltr_model_retrained.pkl  ·  Spearman ρ=0.25\n"
        "24 features: skill overlap · trajectory · behavioral",
        "LTR MODEL", C_STAGE2, width=4.2, height=1.5,
    ).move_to(LEFT * 1.6 + DOWN * 3.35)

    six_dim = component_box(
        "6-Dim Scoring",
        "Tech · Traj · Domain\nEngage · Fit · Bonus",
        "SCORE", C_STAGE2, width=3.0, height=1.4,
    ).move_to(RIGHT * 3.5 + DOWN * 3.35)

    # ── Stage 3 banner ───────────────────────────────────────────────────────
    s3_banner = stage_banner(
        "STAGE 3 — LLM Reasoning", C_STAGE3,
    ).move_to(DOWN * 4.32)

    # ── Row 5: LLM reasoning ─────────────────────────────────────────────────
    llm_full = component_box(
        "Qwen2.5-3B-Instruct",
        "top-50 listwise  ·  per-candidate reasoning\nactual skills + company + gap",
        "LLM FULL", C_STAGE3, width=3.8, height=1.45,
    ).move_to(LEFT * 2.4 + DOWN * 5.18)

    template = component_box(
        "Template Summaries",
        "ranks 51–100  ·  [Tech:X | Traj:X | …]\ndimension-score prefix",
        "TEMPLATE", C_STAGE3, width=3.6, height=1.4,
    ).move_to(RIGHT * 2.4 + DOWN * 5.18)

    # ── Row 6: Output ────────────────────────────────────────────────────────
    output = component_box(
        "submission_v4.csv",
        "100 candidates  ·  rank · score · reasoning",
        "OUTPUT", C_OUTPUT, width=4.4, tag_color=C_OUTPUT,
    ).move_to(DOWN * 6.5)

    # ── Fairness audit boxes (outside main column) ───────────────────────────
    pool_audit = audit_box(
        "Pool Fairness Audit",
        "fairness_audit.py\nDI @ top-100  ·  JSON export",
    ).move_to(LEFT * 6.0 + DOWN * 1.1)

    output_audit = audit_box(
        "Output Fairness Audit",
        "fairness_audit.py\nDI @ top-10/50/100",
    ).move_to(RIGHT * 6.0 + DOWN * 5.18)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_items = [
        ("Input / Output",             C_INPUT),
        ("Stage 1: Hybrid Retrieval",  C_STAGE1),
        ("Stage 2: LTR Re-ranking",    C_STAGE2),
        ("Stage 3: LLM Reasoning",     C_STAGE3),
        ("Fairness Audit (2 points)",  C_AUDIT),
    ]
    legend = VGroup()
    for lbl, col in legend_items:
        dot = Square(side_length=0.17, fill_color=col, fill_opacity=1, stroke_width=0)
        txt = Text(lbl, font_size=11, color=C_MUTED)
        legend.add(VGroup(dot, txt).arrange(RIGHT, buff=0.14))
    legend.arrange(DOWN, aligned_edge=LEFT, buff=0.11)

    # ── Arrows ───────────────────────────────────────────────────────────────
    kw = dict(buff=0.08, stroke_width=2, color=C_ARROW)

    arrows = VGroup(
        # [0]  JD Input → JD Parser
        Arrow(jd_input.get_bottom(), jd_parser.get_top(), **kw),
        # [1]  Candidates → Dense
        Arrow(candidates.get_bottom(), dense.get_top() + RIGHT * 0.3, **kw),
        # [2]  Candidates → BM25
        Arrow(candidates.get_bottom(), bm25.get_top() + LEFT * 0.3, **kw),
        # [3]  JD Parser → Dense (diagonal down-right)
        Arrow(jd_parser.get_bottom() + RIGHT * 0.4,
              dense.get_top()  + LEFT * 0.6, **kw),
        # [4]  Dense → RRF
        Arrow(dense.get_bottom(), rrf.get_top() + LEFT * 0.5,
              buff=0.08, stroke_width=2, color=C_STAGE1),
        # [5]  BM25 → RRF
        Arrow(bm25.get_bottom(), rrf.get_top() + RIGHT * 0.5,
              buff=0.08, stroke_width=2, color=C_STAGE1),
        # [6]  RRF → CatBoost
        Arrow(rrf.get_bottom(), catboost.get_top(),
              buff=0.08, stroke_width=2.5, color=C_ARROW),
        # [7]  CatBoost → 6-Dim Scoring
        Arrow(catboost.get_right(), six_dim.get_left(),
              buff=0.08, stroke_width=2, color=C_STAGE2),
        # [8]  CatBoost → Qwen
        Arrow(catboost.get_bottom() + LEFT * 0.3, llm_full.get_top(),
              buff=0.08, stroke_width=2, color=C_ARROW),
        # [9]  CatBoost → Template
        Arrow(catboost.get_bottom() + RIGHT * 0.6,
              template.get_top() + LEFT * 0.3,
              buff=0.08, stroke_width=2, color=C_ARROW),
        # [10] Qwen → Output
        Arrow(llm_full.get_bottom(), output.get_top() + LEFT * 0.6,
              buff=0.08, stroke_width=2.5, color=C_OUTPUT),
        # [11] Template → Output
        Arrow(template.get_bottom(), output.get_top() + RIGHT * 0.4,
              buff=0.08, stroke_width=2.5, color=C_OUTPUT),
    )

    # ── Dashed audit connectors ───────────────────────────────────────────────
    pool_arrow = DashedLine(
        rrf.get_left(), pool_audit.get_right(),
        color=C_AUDIT, stroke_width=2, dash_length=0.09,
    )
    pool_lbl = Text("pool audit", font_size=10, color=C_AUDIT).next_to(
        pool_arrow, UP, buff=0.06)

    out_arrow = DashedLine(
        template.get_right(), output_audit.get_left(),
        color=C_AUDIT, stroke_width=2, dash_length=0.09,
    )
    out_lbl = Text("output audit", font_size=10, color=C_AUDIT).next_to(
        out_arrow, UP, buff=0.06)

    # ── Flow labels ───────────────────────────────────────────────────────────
    top500_lbl = Text("top-500 each", font_size=10, color=C_MUTED).next_to(
        rrf.get_top(), RIGHT, buff=0.15)
    top100_lbl = Text("top-100", font_size=10, color=C_MUTED).next_to(
        rrf.get_bottom(), RIGHT, buff=0.15)

    # ── Assemble all content and scale to fit frame ───────────────────────────
    content = VGroup(
        title,
        jd_input, candidates, jd_parser,
        s1_banner, dense, bm25, rrf,
        pool_audit,
        s2_banner, catboost, six_dim,
        s3_banner, llm_full, template,
        output_audit, output,
        arrows, pool_arrow, pool_lbl, out_arrow, out_lbl,
        top500_lbl, top100_lbl,
    )
    content.scale_to_fit_height(7.4)
    content.move_to(ORIGIN + DOWN * 0.1)

    # Legend is placed AFTER scaling so it fits in a corner without interference
    legend.scale(0.85).to_corner(DR, buff=0.22)

    return dict(
        title=title, jd_input=jd_input, candidates=candidates,
        jd_parser=jd_parser,
        s1_banner=s1_banner, dense=dense, bm25=bm25, rrf=rrf,
        pool_audit=pool_audit,
        s2_banner=s2_banner, catboost=catboost, six_dim=six_dim,
        s3_banner=s3_banner, llm_full=llm_full, template=template,
        output_audit=output_audit, output=output,
        legend=legend, arrows=arrows,
        pool_arrow=pool_arrow, pool_lbl=pool_lbl,
        out_arrow=out_arrow, out_lbl=out_lbl,
        top500_lbl=top500_lbl, top100_lbl=top100_lbl,
        all=VGroup(content, legend),
    )


# ── Animated scene ────────────────────────────────────────────────────────────

class ArchitectureDiagram(Scene):
    def construct(self):
        self.camera.background_color = BG
        d = build_diagram()

        # Title
        self.play(FadeIn(d["title"], shift=DOWN * 0.15), run_time=0.7)
        self.wait(0.15)

        # Inputs
        self.play(
            LaggedStart(
                FadeIn(d["jd_input"],   shift=RIGHT * 0.2),
                FadeIn(d["candidates"], shift=LEFT  * 0.2),
                lag_ratio=0.4,
            ), run_time=0.8,
        )
        self.play(GrowArrow(d["arrows"][0]), run_time=0.5)
        self.play(FadeIn(d["jd_parser"], shift=UP * 0.12), run_time=0.6)

        # Stage 1 — Hybrid Retrieval
        self.play(FadeIn(d["s1_banner"]), run_time=0.5)
        self.play(
            LaggedStart(
                FadeIn(d["dense"], shift=UP * 0.15),
                FadeIn(d["bm25"],  shift=UP * 0.15),
                lag_ratio=0.35,
            ), run_time=0.8,
        )
        self.play(
            GrowArrow(d["arrows"][1]),
            GrowArrow(d["arrows"][2]),
            GrowArrow(d["arrows"][3]),
            run_time=1.0,
        )
        self.play(
            GrowArrow(d["arrows"][4]),
            GrowArrow(d["arrows"][5]),
            FadeIn(d["top500_lbl"]),
            run_time=0.8,
        )
        self.play(FadeIn(d["rrf"], scale=0.95), run_time=0.6)

        # Pool fairness audit
        self.play(
            FadeIn(d["pool_audit"], shift=RIGHT * 0.15),
            Create(d["pool_arrow"]),
            FadeIn(d["pool_lbl"]),
            run_time=0.8,
        )

        # Stage 2 — CatBoost LTR
        self.play(FadeIn(d["s2_banner"]), run_time=0.5)
        self.play(
            GrowArrow(d["arrows"][6]),
            FadeIn(d["top100_lbl"]),
            run_time=0.6,
        )
        self.play(FadeIn(d["catboost"], shift=UP * 0.12), run_time=0.6)
        self.play(GrowArrow(d["arrows"][7]), run_time=0.5)
        self.play(FadeIn(d["six_dim"], shift=LEFT * 0.12), run_time=0.5)

        # Stage 3 — LLM Reasoning
        self.play(FadeIn(d["s3_banner"]), run_time=0.5)
        self.play(
            GrowArrow(d["arrows"][8]),
            GrowArrow(d["arrows"][9]),
            run_time=0.7,
        )
        self.play(
            LaggedStart(
                FadeIn(d["llm_full"], shift=UP * 0.12),
                FadeIn(d["template"], shift=UP * 0.12),
                lag_ratio=0.4,
            ), run_time=0.8,
        )

        # Output fairness audit
        self.play(
            FadeIn(d["output_audit"], shift=LEFT * 0.15),
            Create(d["out_arrow"]),
            FadeIn(d["out_lbl"]),
            run_time=0.7,
        )

        # Output CSV
        self.play(
            GrowArrow(d["arrows"][10]),
            GrowArrow(d["arrows"][11]),
            run_time=0.7,
        )
        self.play(FadeIn(d["output"], scale=0.93), run_time=0.6)

        # Legend
        self.play(FadeIn(d["legend"], shift=UP * 0.12), run_time=0.6)
        self.wait(2.5)


# ── Static scene (PNG export) ─────────────────────────────────────────────────

class ArchitectureDiagramStatic(Scene):
    """Full diagram in one frame — render with -s for PNG."""

    def construct(self):
        self.camera.background_color = BG
        d = build_diagram()
        self.add(d["all"])
        self.wait(0.1)
