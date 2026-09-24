"""
AI PCB Engineering Reviewer
----------------------------
A Streamlit web app that lets a hardware engineer upload a schematic PDF
and get a deep, structured engineering review from a Google Gemini
multimodal model, focused on:
    1. Signal Integrity (SI)
    2. Power / Ground loop issues
    3. Common-mode coupling risks

How it works (high level):
    1. User uploads a schematic PDF.
    2. We render every page of the PDF into a high-resolution image
       (schematics are visual, so the AI needs to *see* them, not just
       read text).
    3. Each page image is base64-encoded and sent to the Gemini vision
       model along with a detailed engineering-review prompt.
    4. The model's markdown report is parsed into sections and shown
       in a clean, tabbed dashboard. The full report can be downloaded.

Deployment:
    - requirements.txt lists the needed packages.
    - Push app.py + requirements.txt to a GitHub repo and deploy on
      Streamlit Community Cloud (or any Streamlit-compatible host).
    - The user supplies their own Google Gemini API key at runtime, so no
      secret ever needs to be committed to the repo.
"""

import io
import re

import fitz  # PyMuPDF - used to render PDF pages as images
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Gemini model configuration.
DEFAULT_MODEL = "gemini-3.7-flash"
MODEL_OPTIONS = [
    "gemini-3.7-flash",
]

# Section headers we ask the model to use, so we can split the report
# into neat tabs afterwards. Keep these EXACT strings in sync with the
# prompt in build_review_prompt().
REPORT_SECTIONS = [
    "Executive Summary",
    "Signal Integrity Analysis",
    "Power and Ground Loop Analysis",
    "Common Mode Coupling Analysis",
    "Prioritized Recommendations",
]


# ---------------------------------------------------------------------------
# PDF -> Image helpers
# ---------------------------------------------------------------------------

def render_pdf_pages_to_images(pdf_bytes: bytes, zoom: float = 2.0) -> list[Image.Image]:
    """Convert every page of an uploaded PDF into a PIL Image.

    We render at a higher zoom factor (default 2x ~ 144 DPI) so fine
    schematic details (reference designators, trace labels, small text)
    stay legible to the vision model.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.
        zoom: Scale factor applied to the default 72 DPI PDF resolution.

    Returns:
        A list of PIL Image objects, one per PDF page, in page order.
    """
    images = []
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix)
            img_bytes = pixmap.tobytes("png")
            images.append(Image.open(io.BytesIO(img_bytes)))
    return images


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_review_prompt(extra_notes: str) -> str:
    """Build the detailed engineering-review instruction sent to Gemini.

    Args:
        extra_notes: Optional free-text context the user supplied about
            the design (e.g. "4-layer board, switching regulator on
            page 2, high-speed USB 3.0 lines").

    Returns:
        The full instruction string for the model.
    """
    notes_block = f"\nAdditional design context from the user:\n{extra_notes}\n" if extra_notes else ""

    return f"""
You are a senior PCB hardware design engineer performing a rigorous schematic
review. You are shown one or more schematic pages (as images) from a single
PCB design. Study every net, component value, reference designator, and
connector you can identify.
{notes_block}
Produce a deep-dive engineering report in MARKDOWN using EXACTLY these five
'##' headings, in this order, and nothing else before or after them:

## Executive Summary
A short (4-6 sentence) plain-language overview of the design's overall
health and the single biggest risk found.

## Signal Integrity Analysis
Identify and explain concerns such as: high-speed / clock / differential
pairs lacking series or termination resistors, impedance-sensitive nets
without a clear reference plane, long unbuffered traces implied by the
schematic, missing decoupling near high-speed ICs, fan-out or routing
choices visible in the schematic that risk reflections or crosstalk,
and any single-ended signals that should likely be differential.

## Power and Ground Loop Analysis
Identify and explain concerns such as: decoupling capacitor placement and
values versus IC power pins, missing bulk/bypass capacitance, star-grounding
vs multi-point grounding conflicts, ground return path length for
high-current or high-speed loops, split/isolated ground schemes and their
stitching, power sequencing risks, and any large physical loop areas
implied by how power and return nets are drawn.

## Common Mode Coupling Analysis
Identify and explain concerns such as: cable/connector shield grounding,
common-mode choke usage (or absence) on I/O and power lines, isolation
barrier crossings, unbalanced differential routing that could convert
differential noise to common-mode, and proximity of noisy switching nets
to sensitive analog or shield references.

## Prioritized Recommendations
A numbered list (highest impact first) of concrete, actionable fixes. For
each item state: the issue, the risk if unresolved, and the specific
schematic-level fix (e.g. component to add, net to reroute, value to
change).

Formatting rules:
- Reference specific component designators, net names, or page numbers
  whenever you can see them in the image.
- If a page's image quality or resolution prevents you from confirming a
  detail, say so explicitly rather than guessing silently.
- Be direct and technical; this report is for a hardware engineer, not a
  general audience.
""".strip()


def build_gemini_contents(images: list[Image.Image], extra_notes: str) -> list:
    """Build Gemini multimodal contents from rendered schematic pages."""
    contents = []
    for img in images:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        contents.append(
            types.Part.from_bytes(
                data=buffer.getvalue(),
                mime_type="image/png",
            )
        )
    contents.append(build_review_prompt(extra_notes))
    return contents


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------

def call_gemini(api_key: str, model: str, contents: list) -> str:
    """Send the multimodal PCB review request to Google Gemini."""
    if not api_key:
        raise RuntimeError("Gemini API key is missing.")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an expert PCB and signal-integrity engineer with "
                    "20+ years of hardware review experience. Give precise, "
                    "actionable, technically grounded feedback. Analyze only "
                    "what can reasonably be established from the supplied "
                    "schematic images and clearly mark uncertainty."
                ),
                temperature=0.2,
                max_output_tokens=6000,
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")
        return response.text
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Report parsing / display helpers
# ---------------------------------------------------------------------------

def split_report_into_sections(report_text: str) -> dict[str, str]:
    """Split a markdown report into a dict keyed by REPORT_SECTIONS titles.

    Falls back gracefully: any heading the model produced that isn't in
    REPORT_SECTIONS is bundled under 'Other Notes' so no content is lost.
    """
    sections: dict[str, str] = {}
    # Split on '## Heading' lines, keeping the heading text.
    parts = re.split(r"\n(?=##\s+)", report_text.strip())
    for part in parts:
        match = re.match(r"##\s+(.*?)\n(.*)", part.strip(), re.DOTALL)
        if match:
            title, body = match.group(1).strip(), match.group(2).strip()
            sections[title] = body
        elif part.strip():
            sections.setdefault("Other Notes", "")
            sections["Other Notes"] += part.strip() + "\n"
    return sections


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

from io import BytesIO
from xml.sax.saxutils import escape

def configure_page():
    st.set_page_config(
        page_title="PCB/Schematic Reviewer",
        page_icon="🟩",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --green: #39ff14;
            --green2: #9aff75;
            --black: #030504;
            --panel: #0b0e0c;
            --panel2: #111512;
            --grey: #555d58;
            --text: #e9f7e8;
            --muted: #8b978d;
            --line: rgba(57,255,20,.24);
        }

        html, body, [class*="css"] {
            font-family: Inter, sans-serif;
        }

        .stApp {
            color: var(--text);
            background:
                linear-gradient(rgba(57,255,20,.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(57,255,20,.035) 1px, transparent 1px),
                radial-gradient(circle at 50% -10%, rgba(57,255,20,.09), transparent 42%),
                var(--black);
            background-size: 34px 34px, 34px 34px, auto, auto;
        }

        [data-testid="stHeader"] { background: transparent; }

        .block-container {
            max-width: 1440px;
            padding-top: 1.7rem;
            padding-bottom: 4rem;
        }

        /* PCB traces */
        .pcb-board {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 2.25rem;
            margin-bottom: 1.25rem;
            background:
                linear-gradient(90deg, transparent 49%, rgba(57,255,20,.06) 50%, transparent 51%),
                linear-gradient(transparent 49%, rgba(57,255,20,.06) 50%, transparent 51%),
                radial-gradient(circle at 82% 22%, rgba(57,255,20,.12), transparent 24%),
                #070a07;
            background-size: 42px 42px, 42px 42px, auto, auto;
            box-shadow: inset 0 0 55px rgba(57,255,20,.035), 0 22px 80px rgba(0,0,0,.45);
        }

        .pcb-board:before, .pcb-board:after {
            content: "";
            position: absolute;
            pointer-events: none;
            border: 1px solid rgba(57,255,20,.18);
            opacity: .8;
        }
        .pcb-board:before {
            width: 180px; height: 70px; right: 7%; top: 24%;
            border-left: 0; border-bottom: 0; border-radius: 0 18px 0 0;
        }
        .pcb-board:after {
            width: 120px; height: 45px; left: 6%; bottom: 16%;
            border-right: 0; border-top: 0; border-radius: 0 0 0 14px;
        }

        .eyebrow {
            color: var(--green);
            font-family: "JetBrains Mono", monospace;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .13em;
            text-transform: uppercase;
        }

        .pcb-board h1 {
            font-size: clamp(2.2rem, 5vw, 4.35rem);
            line-height: .98;
            letter-spacing: -.055em;
            margin: .55rem 0 .8rem;
            color: #fff;
        }

        .pcb-board h1 span { color: var(--green); text-shadow: 0 0 28px rgba(57,255,20,.25); }

        .hero-copy {
            max-width: 800px;
            color: #a9b5ab;
            font-size: 1.04rem;
            line-height: 1.7;
        }

        .trace-status {
            display: inline-flex;
            align-items: center;
            gap: .5rem;
            margin-top: 1rem;
            padding: .5rem .75rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--green2);
            background: rgba(57,255,20,.045);
            font-family: "JetBrains Mono", monospace;
            font-size: .76rem;
        }
        .led { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 12px var(--green); }

        .workflow {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: .8rem;
            margin: 1rem 0 1.25rem;
        }
        .step {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: rgba(10,14,10,.82);
        }
        .step-num {
            display: inline-grid; place-items: center;
            width: 30px; height: 30px; border-radius: 50%;
            background: var(--green); color: #000;
            font-weight: 900; margin-right: .6rem;
        }
        .step b { color: #fff; }
        .step small { display:block; color: var(--muted); margin-top:.35rem; }

        .feature-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: .85rem;
            margin: 1rem 0 1.35rem;
        }
        .feature {
            border: 1px solid var(--line);
            border-radius: 17px;
            padding: 1.05rem;
            background: linear-gradient(145deg, rgba(14,19,14,.96), rgba(6,9,6,.86));
            transition: transform .2s ease, border-color .2s ease;
        }
        .feature:hover { transform: translateY(-2px); border-color: rgba(57,255,20,.55); }
        .feature .icon { font-size: 1.5rem; }
        .feature b { display:block; margin:.45rem 0 .25rem; color: var(--green2); }
        .feature span { color: var(--muted); font-size:.82rem; line-height:1.5; }

        .section-label {
            color: var(--green);
            font-family: "JetBrains Mono", monospace;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin: 1.45rem 0 .7rem;
        }

        .upload-shell {
            border: 1px dashed rgba(57,255,20,.48);
            border-radius: 18px;
            padding: .8rem;
            background: rgba(57,255,20,.025);
        }

        [data-testid="stFileUploaderDropzone"] {
            background: #151915 !important;
            border-radius: 14px !important;
        }
        [data-testid="stFileUploaderDropzone"] * { color: #dbe5dc !important; }

        textarea, input {
            background: #505651 !important;
            color: #fff !important;
            border: 1px solid rgba(255,255,255,.14) !important;
            border-radius: 11px !important;
        }
        textarea::placeholder, input::placeholder { color: #d2d8d3 !important; }

        .status-card, .metric-card, .report-shell {
            border: 1px solid var(--line);
            border-radius: 17px;
            background: rgba(10,14,10,.9);
        }
        .status-card { padding: 1rem 1.1rem; }
        .metric-card { padding: 1rem 1.1rem; }
        .metric-number { font-size: 1.55rem; font-weight: 800; color: var(--green); }
        .metric-label { color: var(--muted); font-size: .78rem; margin-top:.15rem; }

        .stButton > button, .stDownloadButton > button {
            background: var(--green) !important;
            color: #000 !important;
            border: 1px solid var(--green) !important;
            border-radius: 11px !important;
            font-weight: 900 !important;
            min-height: 2.9rem;
            box-shadow: 0 0 22px rgba(57,255,20,.12);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: #72ff55 !important;
            border-color: #72ff55 !important;
            color: #000 !important;
        }

        .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            background: #0e120f;
            color: #b8c4b9;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(57,255,20,.1);
            color: var(--green) !important;
        }

        .report-shell { padding: 1.25rem; }
        .report-title {
            font-size: 1.8rem; font-weight: 800; color: #fff; margin-bottom:.25rem;
        }
        .report-subtitle {
            color: var(--muted); font-family:"JetBrains Mono",monospace; font-size:.76rem;
        }
        .small-note { color: var(--muted); font-size: .76rem; line-height:1.55; }

        @media (max-width: 800px) {
            .feature-row, .workflow { grid-template-columns: 1fr; }
            .pcb-board { padding: 1.35rem; }
            .block-container { padding-top: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric(number: str, label: str):
    st.markdown(
        f'<div class="metric-card"><div class="metric-number">{number}</div><div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def make_pdf(report_text: str, filename: str = "pcb_schematic_review.pdf") -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
    from reportlab.lib.units import mm

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("PCBTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, textColor=colors.HexColor("#111111"), spaceAfter=10)
    body = ParagraphStyle("PCBBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=14, textColor=colors.HexColor("#222222"), spaceAfter=7)
    heading = ParagraphStyle("PCBHeading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, textColor=colors.HexColor("#111111"), spaceBefore=10, spaceAfter=6)

    story = [Paragraph("PCB/Schematic Reviewer — Engineering Report", title),
             Paragraph("AI-assisted engineering review", body),
             Spacer(1, 5)]

    for raw in report_text.splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 4))
        elif line.startswith("#"):
            text = re.sub(r"^#+\s*", "", line)
            story.append(Paragraph(escape(text), heading))
        elif line.startswith(("- ", "* ")):
            story.append(Paragraph("• " + escape(line[2:]), body))
        else:
            story.append(Paragraph(escape(line), body))

    doc.build(story)
    return buf.getvalue()


def make_docx(report_text: str) -> bytes:
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_heading("PCB/Schematic Reviewer — Engineering Report", level=0)
    p = doc.add_paragraph("AI-assisted engineering review")
    p.runs[0].bold = True

    for raw in report_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 3)
            doc.add_heading(re.sub(r"^#+\s*", "", line), level=level)
        elif line.startswith(("- ", "* ")):
            doc.add_paragraph(line[2:], style="List Bullet")
        else:
            doc.add_paragraph(line)

    for p in doc.paragraphs:
        for run in p.runs:
            run.font.name = "Arial"
            run.font.size = Pt(10)

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def render_empty_state():
    st.markdown(
        """
        <div class="status-card" style="text-align:center; padding:2.4rem 1.5rem;">
            <div style="font-size:2.5rem;">🟩</div>
            <h3 style="margin:.5rem 0; color:#fff;">Your engineering review starts here</h3>
            <div style="color:#8b978d; max-width:650px; margin:auto;">
                Add a schematic PDF, optionally tell us about the design, and let the reviewer
                surface the highest-value engineering risks.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    configure_page()
    # Frontend has no sidebar. Configuration is handled internally.
    settings = {"api_key": st.secrets.get("GEMINI_API_KEY", None), "model": DEFAULT_MODEL, "zoom": 2.0}

    st.markdown(
        '<div class="trace-status" style="margin-bottom:1rem;"><span class="led"></span> READY • SETTINGS MANAGED SECURELY</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="pcb-board">
            <div class="eyebrow">PCB / SCHEMATIC REVIEWER</div>
            <h1>Review your design.<br><span>Catch risks earlier.</span></h1>
            <div class="hero-copy">
                AI-assisted engineering analysis for signal integrity, power & ground,
                and EMI/common-mode coupling — presented as practical, prioritized findings.
            </div>
            <div class="trace-status"><span class="led"></span> ANALYSIS CONSOLE ONLINE</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="workflow">
            <div class="step"><span class="step-num">1</span><b>Add schematic</b><small>Upload your PDF and optionally add design context.</small></div>
            <div class="step"><span class="step-num">2</span><b>Analyze PCB</b><small>Run the AI review and inspect prioritized engineering findings.</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-row">
            <div class="feature"><div class="icon">⚡</div><b>Signal Integrity</b><span>Termination, reflections, interfaces and high-speed risk areas.</span></div>
            <div class="feature"><div class="icon">🔋</div><b>Power & Ground</b><span>Decoupling, return paths, loops and power integrity concerns.</span></div>
            <div class="feature"><div class="icon">📡</div><b>EMI & Coupling</b><span>Common-mode paths, filtering, shields and coupling risks.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">01 · Add schematic</div>', unsafe_allow_html=True)

    upload_col, context_col = st.columns([1.3, 1], gap="large")
    with upload_col:
        st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
        uploaded_pdf = st.file_uploader(
            "📄 Drag & drop your schematic PDF",
            type=["pdf"],
            help="Multi-page schematic PDFs are supported.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_pdf:
            st.success(f"✓ {uploaded_pdf.name} loaded • {uploaded_pdf.size / 1024:.0f} KB", icon="🟩")

    with context_col:
        extra_notes = st.text_area(
            "Design context (optional)",
            placeholder="Example: 4-layer board • USB 3.x • buck regulator on page 2 • sensitive ADC on page 4",
            height=132,
        )

    if uploaded_pdf:
        st.markdown('<div class="section-label">02 · Analyze PCB</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: render_metric("✓", "Schematic ready")
        with c2: render_metric(f"{settings['zoom']:.1f}×", "Render quality")
        with c3: render_metric("AI", "Engineering review")

        st.write("")
        run_clicked = st.button(
            "⚡ Analyze My PCB/Schematic",
            type="primary",
            use_container_width=True,
        )
    else:
        run_clicked = False
        render_empty_state()

    if run_clicked:
        if not settings["api_key"]:
            st.error("GEMINI_API_KEY is missing from Streamlit Secrets. Add it before running the review.")
            return

        pdf_bytes = uploaded_pdf.getvalue()

        with st.status("Preparing schematic…", expanded=True) as status:
            st.write("Rendering schematic pages.")
            images = render_pdf_pages_to_images(pdf_bytes, zoom=settings["zoom"])
            st.write(f"✓ {len(images)} page(s) prepared")
            status.update(label="Schematic ready", state="complete", expanded=False)

        with st.expander(f"👁 Preview submitted schematic • {len(images)} page(s)", expanded=False):
            cols = st.columns(min(len(images), 4) or 1)
            for i, img in enumerate(images):
                cols[i % len(cols)].image(img, caption=f"Page {i + 1}", use_container_width=True)

        try:
            with st.status("🤖 Analyzing your PCB/schematic…", expanded=True) as status:
                st.write("Checking signal integrity, power/ground, and EMI & coupling.")
                contents = build_gemini_contents(images, extra_notes)
                report_text = call_gemini(settings["api_key"], settings["model"], contents)
                status.update(label="Analysis complete", state="complete", expanded=False)
        except RuntimeError as exc:
            st.error(f"Analysis failed: {exc}")
            return

        st.session_state["last_report"] = report_text

    if "last_report" in st.session_state:
        st.markdown('<div class="section-label">Engineering report</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="report-shell">
                <div class="report-title">PCB/Schematic Review Report</div>
                <div class="report-subtitle">AI-ASSISTED • PRIORITIZED ENGINEERING FINDINGS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        sections = split_report_into_sections(st.session_state["last_report"])
        tab_titles = [t for t in REPORT_SECTIONS if t in sections] + [t for t in sections if t not in REPORT_SECTIONS]
        tabs = st.tabs([f"  {t}" for t in tab_titles])

        for tab, title in zip(tabs, tab_titles):
            with tab:
                st.markdown('<div class="report-shell">', unsafe_allow_html=True)
                st.markdown(sections[title])
                st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        pdf_bytes = make_pdf(st.session_state["last_report"])
        docx_bytes = make_docx(st.session_state["last_report"])

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Download PDF report",
                data=pdf_bytes,
                file_name="pcb_schematic_review.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇️ Download Word report",
                data=docx_bytes,
                file_name="pcb_schematic_review.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
