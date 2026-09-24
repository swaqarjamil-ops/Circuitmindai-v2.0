"""
CircuitMind AI - PCB/Schematic Engineering Reviewer

Streamlit application for reviewing schematic PDFs with Google Gemini.

Key features:
- PDF pages rendered as high-resolution images
- Multimodal Gemini schematic analysis
- Automatic retry for temporary Gemini errors
- Exponential backoff with jitter
- Automatic model fallback
- Streamlit-friendly error handling
- Markdown report tabs
- PDF and Word report downloads
"""

import io
import random
import re
import time
from io import BytesIO
from xml.sax.saxutils import escape

import fitz
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3.8-flash"

# Primary model followed by fallback models.
MODEL_OPTIONS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]

MAX_MODEL_RETRIES = 3
INITIAL_RETRY_DELAY = 2.0
RETRY_JITTER_MAX = 1.0

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

def render_pdf_pages_to_images(
    pdf_bytes: bytes,
    zoom: float = 2.0,
) -> list[Image.Image]:
    """Convert every PDF page into a PIL image."""
    images = []
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img_bytes = pixmap.tobytes("png")
            images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

    return images


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_review_prompt(extra_notes: str) -> str:
    """Build the engineering-review prompt sent to Gemini."""

    notes_block = (
        f"\nAdditional design context from the user:\n{extra_notes}\n"
        if extra_notes
        else ""
    )

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


def build_gemini_contents(
    images: list[Image.Image],
    extra_notes: str,
) -> list:
    """Build Gemini multimodal contents from schematic page images."""

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
# Gemini error handling
# ---------------------------------------------------------------------------

def is_temporary_gemini_error(exc: Exception) -> bool:
    """
    Determine whether a Gemini exception is likely temporary.

    Retries common transient HTTP/API conditions:
    408, 429, 500, 502, 503, 504 and common Gemini status messages.
    """

    error_text = str(exc).upper()

    temporary_markers = [
        "408",
        "429",
        "500",
        "502",
        "503",
        "504",
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "INTERNAL",
        "DEADLINE_EXCEEDED",
        "SERVICE_UNAVAILABLE",
        "TEMPORARILY UNAVAILABLE",
        "HIGH DEMAND",
        "OVERLOADED",
    ]

    return any(marker in error_text for marker in temporary_markers)


def get_gemini_error_message(exc: Exception) -> str:
    """Convert a raw Gemini exception into a concise user-facing message."""

    error_text = str(exc)
    upper_error = error_text.upper()

    if "503" in upper_error or "UNAVAILABLE" in upper_error:
        return "Gemini is temporarily overloaded or unavailable."

    if "429" in upper_error or "RESOURCE_EXHAUSTED" in upper_error:
        return "Gemini rate or quota limit was reached."

    if "401" in upper_error or "403" in upper_error:
        return "Gemini API authentication or permission failed."

    if "400" in upper_error:
        return "Gemini rejected the request. Check the request parameters."

    if "404" in upper_error:
        return (
            "The requested Gemini model was not found or is not available "
            "to this API key."
        )

    if "504" in upper_error or "DEADLINE_EXCEEDED" in upper_error:
        return "Gemini took too long to respond."

    return "Gemini returned an unexpected API error."


# ---------------------------------------------------------------------------
# Gemini API call with retry + fallback
# ---------------------------------------------------------------------------

def call_gemini(
    api_key: str,
    model: str,
    contents: list,
) -> str:
    """
    Send the multimodal PCB review request to Gemini.

    Behavior:
    1. Try the requested primary model.
    2. Retry temporary failures up to MAX_MODEL_RETRIES.
    3. Use exponential backoff with jitter.
    4. Move automatically to the next fallback model.
    5. Stop immediately for permanent API errors.
    """

    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY to "
            "Streamlit Secrets."
        )

    # Requested model first, then remaining configured fallbacks.
    models_to_try = [model]

    for fallback_model in MODEL_OPTIONS:
        if fallback_model not in models_to_try:
            models_to_try.append(fallback_model)

    client = genai.Client(api_key=api_key)
    last_error = None
    total_models = len(models_to_try)

    for model_index, current_model in enumerate(models_to_try):

        if model_index == 0:
            st.write(f"Primary Gemini model: `{current_model}`")
        else:
            st.warning(
                f"Switching to fallback model "
                f"`{current_model}` ({model_index + 1}/{total_models})."
            )

        for attempt in range(1, MAX_MODEL_RETRIES + 1):

            try:
                if attempt == 1:
                    st.write(f"AI analysis using `{current_model}`...")
                else:
                    st.write(
                        f"Retry {attempt}/{MAX_MODEL_RETRIES} "
                        f"using `{current_model}`..."
                    )

                response = client.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "You are an expert PCB and signal-integrity "
                            "engineer with 20+ years of hardware review "
                            "experience. Give precise, actionable, "
                            "technically grounded feedback. Analyze only "
                            "what can reasonably be established from the "
                            "supplied schematic images and clearly mark "
                            "uncertainty."
                        ),
                        thinking_config=types.ThinkingConfig(
                            thinking_level="medium"
                        ),
                        max_output_tokens=6000,
                    ),
                )

                if response is None:
                    raise RuntimeError("Gemini returned no response.")

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                if model_index == 0:
                    st.success(
                        f"Analysis completed using `{current_model}`."
                    )
                else:
                    st.success(
                        f"Analysis completed using fallback model "
                        f"`{current_model}`."
                    )

                return response.text

            except Exception as exc:
                last_error = exc

                # Do not retry permanent errors.
                if not is_temporary_gemini_error(exc):
                    friendly_message = get_gemini_error_message(exc)

                    raise RuntimeError(
                        f"{friendly_message}\n\n"
                        f"Technical details: {exc}"
                    ) from exc

                # Temporary error: retry this model if attempts remain.
                if attempt < MAX_MODEL_RETRIES:

                    delay = INITIAL_RETRY_DELAY ** attempt
                    jitter = random.uniform(0, RETRY_JITTER_MAX)
                    wait_time = delay + jitter

                    st.warning(
                        f"⚠️ `{current_model}` is temporarily unavailable. "
                        f"{get_gemini_error_message(exc)} "
                        f"Retrying in approximately "
                        f"{wait_time:.1f} seconds..."
                    )

                    time.sleep(wait_time)

                else:
                    st.warning(
                        f"⚠️ `{current_model}` failed after "
                        f"{MAX_MODEL_RETRIES} attempts."
                    )

        if model_index < total_models - 1:
            st.info("Trying the next configured Gemini model...")

    # All models failed.
    if last_error:
        raise RuntimeError(
            "All configured Gemini models are temporarily unavailable.\n\n"
            f"{get_gemini_error_message(last_error)}\n\n"
            "Please wait a few minutes and try the PCB analysis again."
        ) from last_error

    raise RuntimeError("Gemini analysis could not be completed.")


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def split_report_into_sections(report_text: str) -> dict[str, str]:
    """Split a markdown report into sections for Streamlit tabs."""

    sections: dict[str, str] = {}

    parts = re.split(
        r"\n(?=##\s+)",
        report_text.strip(),
    )

    for part in parts:

        match = re.match(
            r"##\s+(.*?)\n(.*)",
            part.strip(),
            re.DOTALL,
        )

        if match:
            title = match.group(1).strip()
            body = match.group(2).strip()
            sections[title] = body

        elif part.strip():
            sections.setdefault("Other Notes", "")
            sections["Other Notes"] += part.strip() + "\n"

    return sections


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def configure_page():
    """Configure Streamlit page and application styling."""

    st.set_page_config(
        page_title="CircuitMind AI",
        page_icon="🟩",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --bg: #07110d;
    --surface: #0d1813;
    --surface-2: #111f18;
    --surface-3: #16271e;
    --primary: #65d46e;
    --primary-soft: #b8f5bd;
    --accent: #2fbf71;
    --text: #ffffff;
    --text-heading: #ffffff;
    --text-body: #f1f7f2;
    --text-secondary: #d5e2d8;
    --text-muted: #b8c8bc;
    --muted: #b8c8bc;
    --border: rgba(101, 212, 110, .30);
    --border-soft: rgba(255,255,255,.14);
}

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
    color: var(--text-body);
}

.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp div {
    color: var(--text-body);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-heading) !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em {
    color: var(--text-body) !important;
}

small,
.stCaption,
[data-testid="stCaptionContainer"] {
    color: var(--text-secondary) !important;
}

.stApp {
    background:
        radial-gradient(circle at 50% -8%, rgba(47,191,113,.12), transparent 34%),
        linear-gradient(180deg, #08120e 0%, #050a07 100%);
}

.block-container {
    max-width: 1320px;
    padding: 2.4rem 2rem 4.5rem;
}

.pcb-board {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 2.35rem 2.5rem;
    margin-bottom: 1.35rem;
    background:
        linear-gradient(135deg, rgba(47,191,113,.10), rgba(13,24,19,.94) 52%),
        #0a140f;
    box-shadow: 0 18px 60px rgba(0,0,0,.30);
}

.pcb-board:before {
    content: "";
    position: absolute;
    inset: 0;
    opacity: .23;
    background:
        linear-gradient(90deg, transparent 49.7%, rgba(101,212,110,.16) 50%, transparent 50.3%),
        linear-gradient(transparent 49.7%, rgba(101,212,110,.12) 50%, transparent 50.3%);
    background-size: 46px 46px;
    pointer-events: none;
}

.eyebrow {
    position: relative;
    color: var(--primary);
    font-family: "JetBrains Mono", monospace;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .13em;
    text-transform: uppercase;
}

.pcb-board h1 {
    position: relative;
    font-size: clamp(2.35rem, 4.4vw, 4rem);
    line-height: 1.04;
    letter-spacing: -.045em;
    font-weight: 800;
    margin: .65rem 0 .85rem;
    color: #f7fbf8;
}

.pcb-board h1 span {
    color: var(--primary);
}

.hero-copy {
    position: relative;
    max-width: 760px;
    color: #b4c2b8;
    font-size: .98rem;
    line-height: 1.7;
}

.trace-status {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    padding: .42rem .72rem;
    margin-top: 1rem;
    border: 1px solid rgba(101,212,110,.25);
    border-radius: 999px;
    background: rgba(101,212,110,.055);
    color: var(--primary-soft);
    font-family: "JetBrains Mono", monospace;
    font-size: .68rem;
    font-weight: 600;
    letter-spacing: .04em;
}

.led {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--primary);
    box-shadow: 0 0 10px rgba(101,212,110,.65);
}

.workflow {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .9rem;
    margin: 0 0 1.35rem;
}

.step {
    border: 1px solid var(--border-soft);
    border-radius: 15px;
    padding: .95rem 1.05rem;
    background: rgba(13,24,19,.78);
}

.step-num {
    display: inline-grid;
    place-items: center;
    width: 28px;
    height: 28px;
    margin-right: .55rem;
    border-radius: 50%;
    background: var(--primary);
    color: #07110d;
    font-size: .78rem;
    font-weight: 800;
}

.step b {
    color: #ffffff !important;
    font-size: .9rem;
}

.step small {
    display: block;
    margin-top: .35rem;
    color: #d0ddd3 !important;
    font-size: .75rem;
    line-height: 1.5;
}

.feature-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .9rem;
    margin: 0 0 1.65rem;
}

.feature {
    min-height: 125px;
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    padding: 1.15rem;
    background: linear-gradient(145deg, rgba(17,31,24,.92), rgba(9,17,13,.94));
    box-shadow: 0 8px 25px rgba(0,0,0,.12);
}

.feature .icon {
    font-size: 1.3rem;
}

.feature b {
    display: block;
    margin: .42rem 0 .28rem;
    color: var(--primary-soft);
    font-size: .88rem;
    font-weight: 700;
}

.feature span {
    color: #d0ddd3 !important;
    font-size: .75rem;
    line-height: 1.55;
}

.section-label {
    color: var(--primary);
    font-family: "JetBrains Mono", monospace;
    font-size: .68rem;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin: 1.5rem 0 .7rem;
}

.upload-shell {
    border: 1px dashed rgba(101,212,110,.36);
    border-radius: 16px;
    padding: .65rem;
    background: rgba(101,212,110,.025);
}

[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #173a28 0%, #10261c 55%, #0c1812 100%) !important;
    border-radius: 13px !important;
    border: 1.5px solid rgba(101,212,110,.62) !important;
    box-shadow: inset 0 0 30px rgba(101,212,110,.045), 0 8px 26px rgba(0,0,0,.18) !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #9be6a0 !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #eef7f0 !important;
    font-size: .84rem !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #65d46e !important;
    color: #07110d !important;
    border: 1px solid #65d46e !important;
    border-radius: 9px !important;
    font-weight: 800 !important;
}

textarea, input {
    background: #26352c !important;
    color: #ffffff !important;
    border: 1px solid rgba(101,212,110,.28) !important;
    border-radius: 10px !important;
    font-size: .86rem !important;
}

textarea::placeholder,
input::placeholder {
    color: #c2ccc5 !important;
    opacity: 1 !important;
}

.status-card,
.metric-card,
.report-shell {
    border: 1px solid var(--border-soft);
    border-radius: 16px;
    background: rgba(12,22,17,.90);
}

.status-card {
    padding: 1.05rem 1.15rem;
}

.metric-card {
    padding: .9rem 1rem;
    min-height: 72px;
}

.metric-number {
    font-size: 1.35rem;
    line-height: 1.1;
    font-weight: 800;
    color: var(--primary);
}

.metric-label {
    margin-top: .28rem;
    color: #d0ddd3 !important;
    font-size: .72rem;
    font-weight: 600;
}

.stButton > button,
.stDownloadButton > button {
    background: var(--primary) !important;
    color: #07110d !important;
    border: 1px solid var(--primary) !important;
    border-radius: 10px !important;
    font-size: .84rem !important;
    font-weight: 800 !important;
    min-height: 2.75rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: .35rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 9px;
    padding: .55rem .8rem;
    background: #0e1812;
    color: #aebbb1;
    font-size: .78rem;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: rgba(101,212,110,.09);
    color: var(--primary) !important;
}

.report-shell {
    padding: 1.2rem;
    color: #f1f7f2 !important;
}

.report-shell p,
.report-shell li {
    color: #f1f7f2 !important;
    line-height: 1.75;
}

.report-shell strong,
.report-shell b {
    color: #ffffff !important;
}

.report-shell h1,
.report-shell h2,
.report-shell h3,
.report-shell h4 {
    color: #ffffff !important;
}

.report-shell h2 {
    border-bottom: 1px solid rgba(101,212,110,.25);
    padding-bottom: .45rem;
}

.report-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #f4f8f5;
}

.report-subtitle {
    color: var(--muted);
    font-family: "JetBrains Mono", monospace;
    font-size: .66rem;
    letter-spacing: .04em;
}

@media (max-width: 800px) {
    .feature-row,
    .workflow {
        grid-template-columns: 1fr;
    }

    .pcb-board {
        padding: 1.55rem;
    }

    .block-container {
        padding: 1.15rem 1rem 3rem;
    }

    .pcb-board h1 {
        font-size: 2.35rem;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def render_metric(number: str, label: str):
    """Render a small status metric card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{number}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Report export helpers
# ---------------------------------------------------------------------------

def make_pdf(
    report_text: str,
    filename: str = "pcb_schematic_review.pdf",
) -> bytes:
    """Generate a PDF engineering report."""

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    buf = BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "PCBTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#111111"),
        spaceAfter=10,
    )

    body = ParagraphStyle(
        "PCBBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#222222"),
        spaceAfter=7,
    )

    heading = ParagraphStyle(
        "PCBHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#111111"),
        spaceBefore=10,
        spaceAfter=6,
    )

    story = [
        Paragraph(
            "PCB/Schematic Reviewer — Engineering Report",
            title,
        ),
        Paragraph(
            "AI-assisted engineering review",
            body,
        ),
        Spacer(1, 5),
    ]

    for raw in report_text.splitlines():

        line = raw.strip()

        if not line:
            story.append(Spacer(1, 4))

        elif line.startswith("#"):
            text = re.sub(r"^#+\s*", "", line)
            story.append(
                Paragraph(
                    escape(text),
                    heading,
                )
            )

        elif line.startswith(("- ", "* ")):
            story.append(
                Paragraph(
                    "• " + escape(line[2:]),
                    body,
                )
            )

        else:
            story.append(
                Paragraph(
                    escape(line),
                    body,
                )
            )

    doc.build(story)

    return buf.getvalue()


def make_docx(report_text: str) -> bytes:
    """Generate a Word engineering report."""

    from docx import Document
    from docx.shared import Pt

    doc = Document()

    doc.add_heading(
        "PCB/Schematic Reviewer — Engineering Report",
        level=0,
    )

    p = doc.add_paragraph(
        "AI-assisted engineering review"
    )
    p.runs[0].bold = True

    for raw in report_text.splitlines():

        line = raw.strip()

        if not line:
            continue

        if line.startswith("#"):

            level = min(
                len(line) - len(line.lstrip("#")),
                3,
            )

            doc.add_heading(
                re.sub(r"^#+\s*", "", line),
                level=level,
            )

        elif line.startswith(("- ", "* ")):

            doc.add_paragraph(
                line[2:],
                style="List Bullet",
            )

        else:

            doc.add_paragraph(line)

    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            run.font.name = "Arial"
            run.font.size = Pt(10)

    buf = BytesIO()
    doc.save(buf)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def render_empty_state():
    """Render the initial empty application state."""

    st.markdown(
        """
        <div class="status-card" style="text-align:center; padding:2.4rem 1.5rem;">
            <div style="font-size:2.5rem;">🟩</div>
            <h3 style="margin:.5rem 0; color:#fff;">
                Your engineering review starts here
            </h3>
            <div style="color:#d5e2d8; max-width:650px; margin:auto;">
                Add a schematic PDF, optionally tell us about the design,
                and let the reviewer surface the highest-value engineering risks.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    """Run the CircuitMind AI Streamlit application."""

    configure_page()

    settings = {
        "api_key": st.secrets.get("GEMINI_API_KEY", None),
        "model": DEFAULT_MODEL,
        "zoom": 2.0,
    }

    st.markdown(
        """
        <section class="pcb-board">
            <div class="eyebrow">CircuitMind AI</div>
            <h1>
                Review your design.<br>
                <span>Catch risks earlier.</span>
            </h1>
            <div class="hero-copy">
                AI-assisted engineering analysis for signal integrity,
                power & ground, and EMI/common-mode coupling —
                presented as practical, prioritized findings.
            </div>
            <div class="trace-status">
                <span class="led"></span>
                ANALYSIS CONSOLE ONLINE
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="workflow">
            <div class="step">
                <span class="step-num">1</span>
                <b>Add schematic</b>
                <small>Upload your PDF and optionally add design context.</small>
            </div>
            <div class="step">
                <span class="step-num">2</span>
                <b>Analyze PCB</b>
                <small>Run the AI review and inspect prioritized engineering findings.</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="feature-row">
            <div class="feature">
                <div class="icon">⚡</div>
                <b>Signal Integrity</b>
                <span>Termination, reflections, interfaces and high-speed risk areas.</span>
            </div>
            <div class="feature">
                <div class="icon">🔋</div>
                <b>Power & Ground</b>
                <span>Decoupling, return paths, loops and power integrity concerns.</span>
            </div>
            <div class="feature">
                <div class="icon">📡</div>
                <b>EMI & Coupling</b>
                <span>Common-mode paths, filtering, shields and coupling risks.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">01 · Add schematic</div>',
        unsafe_allow_html=True,
    )

    upload_col, context_col = st.columns(
        [1.3, 1],
        gap="large",
    )

    with upload_col:

        st.markdown(
            '<div class="upload-shell">',
            unsafe_allow_html=True,
        )

        uploaded_pdf = st.file_uploader(
            "📄 Drag & drop your schematic PDF",
            type=["pdf"],
            help="Multi-page schematic PDFs are supported.",
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        if uploaded_pdf:
            st.success(
                f"✓ {uploaded_pdf.name} loaded • "
                f"{uploaded_pdf.size / 1024:.0f} KB",
                icon="🟩",
            )

    with context_col:

        st.markdown(
            """
            <div style="font-size:.98rem;font-weight:800;color:#f1f7f2;margin:0 0 .35rem;">
                ✦ Design context
                <span style="color:#9eaea3;font-size:.78rem;font-weight:600;">
                    (optional)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        extra_notes = st.text_area(
            "Design context (optional)",
            placeholder=(
                "Example: 4-layer board • USB 3.x • buck regulator on page 2 "
                "• sensitive ADC on page 4"
            ),
            height=132,
            label_visibility="collapsed",
        )

    if uploaded_pdf:

        st.markdown(
            '<div class="section-label">02 · Analyze PCB</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            render_metric("✓", "Schematic ready")

        with c2:
            render_metric(
                f"{settings['zoom']:.1f}×",
                "Render quality",
            )

        with c3:
            render_metric("AI", "Engineering review")

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
            st.error(
                "GEMINI_API_KEY is missing from Streamlit Secrets. "
                "Add it before running the review."
            )
            return

        pdf_bytes = uploaded_pdf.getvalue()

        # ---------------------------------------------------------------
        # PDF rendering
        # ---------------------------------------------------------------

        with st.status(
            "Preparing schematic…",
            expanded=True,
        ) as status:

            try:

                st.write("Rendering schematic pages.")

                images = render_pdf_pages_to_images(
                    pdf_bytes,
                    zoom=settings["zoom"],
                )

                if not images:
                    raise RuntimeError(
                        "The uploaded PDF contains no readable pages."
                    )

                st.write(
                    f"✓ {len(images)} page(s) prepared"
                )

                status.update(
                    label="Schematic ready",
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:

                status.update(
                    label="Schematic preparation failed",
                    state="error",
                    expanded=True,
                )

                st.error(
                    "⚠️ Could not prepare the schematic PDF."
                )

                with st.expander("Technical details"):
                    st.code(str(exc))

                return

        # ---------------------------------------------------------------
        # Schematic preview
        # ---------------------------------------------------------------

        with st.expander(
            f"👁 Preview submitted schematic • "
            f"{len(images)} page(s)",
            expanded=False,
        ):

            cols = st.columns(
                min(len(images), 4) or 1
            )

            for i, img in enumerate(images):

                cols[i % len(cols)].image(
                    img,
                    caption=f"Page {i + 1}",
                    use_container_width=True,
                )

        # ---------------------------------------------------------------
        # Gemini analysis
        # ---------------------------------------------------------------

        status = None

        try:

            with st.status(
                "🤖 Analyzing your PCB/schematic…",
                expanded=True,
            ) as status:

                st.write(
                    "Checking signal integrity, power/ground, "
                    "and EMI & coupling."
                )

                contents = build_gemini_contents(
                    images,
                    extra_notes,
                )

                report_text = call_gemini(
                    settings["api_key"],
                    settings["model"],
                    contents,
                )

                status.update(
                    label="Analysis complete",
                    state="complete",
                    expanded=False,
                )

        except RuntimeError as exc:

            if status is not None:
                status.update(
                    label="Analysis could not be completed",
                    state="error",
                    expanded=True,
                )

            st.error(
                f"⚠️ {exc}"
            )

            st.info(
                "The application automatically retries temporary Gemini "
                "errors and switches to fallback models when possible. "
                "If all models are busy, wait a few minutes and try again."
            )

            return

        except Exception as exc:

            if status is not None:
                status.update(
                    label="Unexpected analysis error",
                    state="error",
                    expanded=False,
                )

            st.error(
                "⚠️ An unexpected error occurred while analyzing "
                "the schematic."
            )

            with st.expander("Technical details"):
                st.code(str(exc))

            return

        st.session_state["last_report"] = report_text

    # -------------------------------------------------------------------
    # Display previous/current report
    # -------------------------------------------------------------------

    if "last_report" in st.session_state:

        st.markdown(
            '<div class="section-label">Engineering report</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="report-shell">
                <div class="report-title">
                    PCB/Schematic Review Report
                </div>
                <div class="report-subtitle">
                    AI-ASSISTED • PRIORITIZED ENGINEERING FINDINGS
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        sections = split_report_into_sections(
            st.session_state["last_report"]
        )

        tab_titles = [
            title
            for title in REPORT_SECTIONS
            if title in sections
        ]

        tab_titles += [
            title
            for title in sections
            if title not in REPORT_SECTIONS
        ]

        if tab_titles:

            tabs = st.tabs(
                [f"  {title}" for title in tab_titles]
            )

            for tab, title in zip(tabs, tab_titles):

                with tab:

                    st.markdown(
                        '<div class="report-shell">',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        sections[title]
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )

        else:

            st.markdown(
                '<div class="report-shell">',
                unsafe_allow_html=True,
            )

            st.markdown(
                st.session_state["last_report"]
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------------
        # Downloads
        # ---------------------------------------------------------------

        st.write("")

        report_pdf = make_pdf(
            st.session_state["last_report"]
        )

        report_docx = make_docx(
            st.session_state["last_report"]
        )

        d1, d2 = st.columns(2)

        with d1:

            st.download_button(
                "⬇️ Download PDF report",
                data=report_pdf,
                file_name="pcb_schematic_review.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with d2:

            st.download_button(
                "⬇️ Download Word report",
                data=report_docx,
                file_name="pcb_schematic_review.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
