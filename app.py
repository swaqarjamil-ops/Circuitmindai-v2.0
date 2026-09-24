"""
CircuitMind AI - PCB/Schematic Engineering Reviewer

Streamlit application for reviewing schematic PDFs with Groq-hosted models.

Primary model:
- openai/gpt-oss-120b on Groq

Fallback models:
- openai/gpt-oss-20b
- qwen/qwen3.8-27b

Important:
GPT-OSS 120B is a text-only model on Groq. The application therefore
extracts the schematic PDF's text/vector labels and sends that engineering
content to GPT-OSS. Qwen 3.8 27B is retained as an optional multimodal
fallback for visual page inspection when needed.
"""

import io
import random
import re
import time
from io import BytesIO
from xml.sax.saxutils import escape

import fitz
import streamlit as st
from groq import Groq
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openai/gpt-oss-120b"

# Primary Groq model followed by fallback models.
MODEL_OPTIONS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

GROQ_API_KEY_NAME = "GROQ_API_KEY"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

MAX_MODEL_RETRIES = 3
INITIAL_RETRY_DELAY = 2.0
RETRY_JITTER_MAX = 1.0
MAX_OUTPUT_TOKENS = 12000

REPORT_SECTIONS = [
    "Executive Summary",
    "Signal Integrity Analysis",
    "Power and Ground Loop Analysis",
    "Common Mode Coupling Analysis",
    "Prioritized Recommendations",
]


# ---------------------------------------------------------------------------
# PDF -> Image/Text helpers
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
            images.append(
                Image.open(io.BytesIO(img_bytes)).convert("RGB")
            )

    return images


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract selectable text, reference designators, net names, labels and
    component values from the schematic PDF.
    """
    page_text = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            page_text.append(
                f"\n===== SCHEMATIC PAGE {page_number} =====\n{text.strip()}"
            )

    combined = "\n".join(page_text).strip()

    # Keep prompts manageable while retaining a large amount of engineering
    # information. GPT-OSS 120B supports a 131K context window on Groq.
    return combined[:500_000]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_review_prompt(
    extra_notes: str,
    extracted_text: str,
) -> str:
    """Build the engineering-review prompt sent to Groq."""

    notes_block = (
        f"\nAdditional design context from the user:\n{extra_notes}\n"
        if extra_notes
        else ""
    )

    text_block = extracted_text or (
        "No selectable PDF text was extracted. Treat component/net details "
        "as unconfirmed and explicitly state that visual inspection is limited."
    )

    return f"""
You are a senior PCB hardware design engineer performing a rigorous schematic
review. Analyze the supplied schematic-derived engineering text carefully.
Use exact reference designators, net names, connector names, IC names and
component values whenever they are present. Do not invent details that are
not supported by the supplied data.
{notes_block}

The source PDF text is provided below. It may contain OCR/vector extraction
artifacts, repeated labels, or incomplete visual information.

{text_block}

Produce a deep-dive engineering report in MARKDOWN using EXACTLY these five
'##' headings, in this order, and nothing else before or after them:

## Executive Summary
A short (4-6 sentence) plain-language overview of the design's overall
health and the single biggest risk found. Clearly distinguish confirmed
findings from items requiring PCB layout verification.

## Signal Integrity Analysis
Identify and explain concerns such as: high-speed / clock / differential
pairs lacking series or termination resistors, impedance-sensitive nets
without a clear reference plane, long unbuffered traces implied by the
schematic, missing decoupling near high-speed ICs, fan-out or routing
choices visible in the schematic that risk reflections or crosstalk,
and single-ended interfaces that should be reviewed for differential
implementation.

## Power and Ground Loop Analysis
Identify and explain concerns such as: decoupling capacitor values and
placement requirements versus IC power pins, missing bulk/bypass
capacitance, star-grounding vs multi-point grounding conflicts, ground
return path risks for high-current or high-speed loops, split/isolated
ground schemes and their stitching, power sequencing risks, and large
physical loop areas implied by power and return topology.

## Common Mode Coupling Analysis
Identify and explain concerns such as: cable/connector shield grounding,
common-mode choke usage or absence on I/O and power lines, isolation
barrier crossings, unbalanced differential interfaces, conversion of
differential noise to common-mode noise, and noisy switching nets near
sensitive analog or shield references.

## Prioritized Recommendations
A numbered list (highest engineering impact first) of concrete,
actionable fixes. For each item state: the issue, the risk if unresolved,
and the specific schematic-level fix. Mark any recommendation that must be
verified against the PCB layout, stack-up, impedance rules, or EMC test data.

Formatting rules:
- Reference specific component designators, net names, or page numbers
  whenever the extracted data supports them.
- Never silently guess a component value, topology, trace length, stack-up,
  impedance, or layout relationship.
- Explicitly say when a conclusion cannot be confirmed from schematic data.
- Be direct and technical; this report is for a hardware engineer.
""".strip()


# ---------------------------------------------------------------------------
# Groq error handling
# ---------------------------------------------------------------------------

def is_temporary_groq_error(exc: Exception) -> bool:
    """Return True for transient Groq/API conditions worth retrying."""

    error_text = str(exc).upper()

    temporary_markers = [
        "408",
        "409",
        "425",
        "429",
        "500",
        "502",
        "503",
        "504",
        "TIMEOUT",
        "TIMED OUT",
        "RATE LIMIT",
        "RATE_LIMIT",
        "TOO MANY REQUESTS",
        "SERVICE UNAVAILABLE",
        "INTERNAL SERVER ERROR",
        "OVERLOADED",
        "TEMPORARILY UNAVAILABLE",
        "CONNECTION RESET",
    ]

    return any(marker in error_text for marker in temporary_markers)


def get_groq_error_message(exc: Exception) -> str:
    """Convert a raw Groq exception into a concise Streamlit message."""

    error_text = str(exc)
    upper_error = error_text.upper()

    if "429" in upper_error or "RATE LIMIT" in upper_error:
        return "Groq rate limit was reached."

    if "401" in upper_error:
        return "Groq API authentication failed. Check GROQ_API_KEY."

    if "403" in upper_error:
        return "Groq API access was denied for this key or model."

    if "404" in upper_error:
        return "The requested Groq model was not found or is unavailable to this API key."

    if any(code in upper_error for code in ("408", "500", "502", "503", "504")):
        return "Groq is temporarily unavailable or the request timed out."

    if "400" in upper_error:
        return "Groq rejected the request. Check model support and request parameters."

    return "Groq returned an unexpected API error."


# ---------------------------------------------------------------------------
# Groq API call with retry + fallback
# ---------------------------------------------------------------------------

def call_groq(
    api_key: str,
    model: str,
    prompt: str,
) -> str:
    """
    Send the PCB engineering review request to Groq.

    The requested model is attempted first. Temporary failures are retried
    with exponential backoff and jitter. If a model remains unavailable,
    the next configured fallback model is tried automatically.
    """

    if not api_key:
        raise RuntimeError(
            "Groq API key is missing. Add GROQ_API_KEY to Streamlit Secrets."
        )

    models_to_try = [model]
    for fallback_model in MODEL_OPTIONS:
        if fallback_model not in models_to_try:
            models_to_try.append(fallback_model)

    client = Groq(
        api_key=api_key,
        base_url=GROQ_BASE_URL,
    )

    last_error = None
    total_models = len(models_to_try)

    for model_index, current_model in enumerate(models_to_try):

        if model_index == 0:
            st.write(f"Primary Groq model: `{current_model}`")
        else:
            st.warning(
                f"Switching to fallback model `{current_model}` "
                f"({model_index + 1}/{total_models})."
            )

        for attempt in range(1, MAX_MODEL_RETRIES + 1):
            try:
                if attempt == 1:
                    st.write(f"AI analysis using `{current_model}`...")
                else:
                    st.write(
                        f"Retry {attempt}/{MAX_MODEL_RETRIES} using "
                        f"`{current_model}`..."
                    )

                response = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert PCB hardware and signal-integrity "
                                "engineer with 20+ years of engineering review "
                                "experience. Give precise, actionable and technically "
                                "grounded feedback. Never invent schematic details."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    reasoning_effort="medium",
                    max_completion_tokens=MAX_OUTPUT_TOKENS,
                    temperature=1.0,
                    stream=False,
                )

                if response is None or not response.choices:
                    raise RuntimeError("Groq returned no response choices.")

                text = response.choices[0].message.content
                if not text or not text.strip():
                    raise RuntimeError("Groq returned an empty response.")

                if model_index == 0:
                    st.success(
                        f"Analysis completed using `{current_model}`."
                    )
                else:
                    st.success(
                        f"Analysis completed using fallback model `{current_model}`."
                    )

                return text.strip()

            except Exception as exc:
                last_error = exc

                if not is_temporary_groq_error(exc):
                    friendly_message = get_groq_error_message(exc)
                    raise RuntimeError(
                        f"{friendly_message}\n\nTechnical details: {exc}"
                    ) from exc

                if attempt < MAX_MODEL_RETRIES:
                    delay = INITIAL_RETRY_DELAY * (2 ** (attempt - 1))
                    jitter = random.uniform(0, RETRY_JITTER_MAX)
                    wait_time = delay + jitter

                    st.warning(
                        f"⚠️ `{current_model}` is temporarily unavailable. "
                        f"{get_groq_error_message(exc)} "
                        f"Retrying in approximately {wait_time:.1f} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    st.warning(
                        f"⚠️ `{current_model}` failed after "
                        f"{MAX_MODEL_RETRIES} attempts."
                    )

        if model_index < total_models - 1:
            st.info("Trying the next configured Groq model...")

    if last_error:
        raise RuntimeError(
            "All configured Groq models were temporarily unavailable.\n\n"
            f"{get_groq_error_message(last_error)}\n\n"
            "Please wait a few minutes and try the PCB analysis again."
        ) from last_error

    raise RuntimeError("Groq analysis could not be completed.")


# ---------------------------------------------------------------------------
# Optional visual fallback using Groq Qwen 3.8
# ---------------------------------------------------------------------------

def analyze_visual_pages_with_qwen(
    api_key: str,
    images: list[Image.Image],
    extra_notes: str,
) -> str:
    """
    Optional visual inspection pass using Groq's qwen/qwen3.8-27b model.

    Qwen 3.8 27B supports image input on Groq. At most three pages are sent
    per request, matching the current Groq vision limit.
    """

    if not api_key or not images:
        return ""

    client = Groq(api_key=api_key, base_url=GROQ_BASE_URL)
    observations = []

    visual_prompt = f"""
You are a PCB schematic visual-inspection assistant. Inspect these schematic
page images and report only visually supported observations useful to a
hardware engineer reviewing signal integrity, power/ground topology, EMI,
and common-mode coupling.

For each visible issue, include page number, reference designator/net name
when readable, observed topology, and why it may matter. Do not invent
values or connections. State clearly when a detail is unreadable.
{('Additional design context: ' + extra_notes) if extra_notes else ''}
""".strip()

    # Groq currently limits Qwen 3.8 to three images per request.
    for start in range(0, len(images), 3):
        batch = images[start:start + 3]
        content = [{"type": "text", "text": visual_prompt}]

        for offset, image in enumerate(batch):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            import base64
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            page_number = start + offset + 1

            content.append(
                {
                    "type": "text",
                    "text": f"SCHEMATIC PAGE {page_number}",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded}"
                    },
                }
            )

        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": content}],
            reasoning_effort="medium",
            max_completion_tokens=5000,
            temperature=1.0,
            stream=False,
        )

        if response.choices and response.choices[0].message.content:
            observations.append(
                f"\n===== VISUAL REVIEW BATCH {start // 3 + 1} =====\n"
                + response.choices[0].message.content.strip()
            )

    return "\n".join(observations).strip()


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------
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
        "api_key": st.secrets.get(GROQ_API_KEY_NAME, None),
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
        # Groq analysis
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

                extracted_text = extract_pdf_text(pdf_bytes)
                st.write(
                    f"✓ Extracted {len(extracted_text):,} characters of schematic text/labels."
                )

                visual_notes = ""
                try:
                    st.write("Performing visual schematic cross-check with Qwen 3.8...")
                    visual_notes = analyze_visual_pages_with_qwen(
                        settings["api_key"],
                        images,
                        extra_notes,
                    )
                    if visual_notes:
                        st.success("Visual schematic cross-check completed.")
                except Exception as visual_exc:
                    st.warning(
                        "Visual cross-check was skipped; continuing with GPT-OSS text analysis. "
                        f"Reason: {get_groq_error_message(visual_exc)}"
                    )

                prompt = build_review_prompt(
                    extra_notes,
                    extracted_text + (
                        "\n\n" + visual_notes if visual_notes else ""
                    ),
                )

                report_text = call_groq(
                    settings["api_key"],
                    settings["model"],
                    prompt,
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
                "The application automatically retries temporary Groq "
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
