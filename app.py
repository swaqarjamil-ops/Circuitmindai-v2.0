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
DEFAULT_MODEL = "gemini-3.6-flash"
MODEL_OPTIONS = [
    "gemini-3.6-flash",
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

def configure_page():
    """Set page-level Streamlit config and inject light custom styling."""
    st.set_page_config(
        page_title="AI PCB Schematic Reviewer — Gemini",
        page_icon="🔬",
        layout="wide",
    )
    # Small CSS polish to make the app feel less like a default form.
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #7C3AED, #06B6D4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .subtitle {
            color: #94A3B8;
            font-size: 1.05rem;
            margin-top: 0.2rem;
        }
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6rem 1.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict:
    """Render the sidebar controls and return the user's chosen settings."""
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Your key is used only for this Streamlit session.",
        )
        model = st.selectbox(
            "Gemini model",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(DEFAULT_MODEL),
            help="Select a Gemini model that supports image input.",
        )
        zoom = st.slider(
            "Rendering quality (zoom)", min_value=1.0, max_value=4.0, value=2.0, step=0.5,
            help="Higher = sharper page images, but slower and more tokens.",
        )
        st.markdown("---")
        st.caption(
            "Built with Streamlit + Google Gemini. Your schematic is sent "
            "directly to the Google Gemini API for analysis and is not stored by "
            "this app."
        )
    return {"api_key": api_key, "model": model, "zoom": zoom}


def main():
    """Entry point: wires together the UI, PDF processing, and API call."""
    configure_page()

    st.markdown('<p class="main-title">🔬 AI PCB Schematic Reviewer — Gemini</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Upload a schematic PDF and get a deep-dive '
        'review on signal integrity, power/ground loops, and common-mode '
        'coupling — powered by Google Gemini.</p>',
        unsafe_allow_html=True,
    )
    st.write("")

    settings = render_sidebar()

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_pdf = st.file_uploader("📄 Upload schematic (PDF)", type=["pdf"])
    with col2:
        extra_notes = st.text_area(
            "Optional design context",
            placeholder="e.g. 4-layer board, DDR4 on page 3, USB-C on page 5...",
            height=100,
        )

    run_clicked = st.button("🚀 Run Deep Analysis", type="primary", disabled=uploaded_pdf is None)

    if run_clicked:
        if not settings["api_key"]:
            st.error("Please enter your Gemini API key in the sidebar first.")
            return

        with st.spinner("Rendering schematic pages..."):
            images = render_pdf_pages_to_images(uploaded_pdf.read(), zoom=settings["zoom"])
            
        st.success(f"Rendered {len(images)} page(s). Sending to Gemini for review...")

        # Show a quick preview strip of the pages being analyzed.
        with st.expander("📎 Preview submitted pages", expanded=False):
            preview_cols = st.columns(min(len(images), 4) or 1)
            for i, img in enumerate(images):
                preview_cols[i % len(preview_cols)].image(img, caption=f"Page {i + 1}", use_container_width=True)

        try:
            with st.spinner("Gemini is analyzing signal integrity, power/ground loops, and coupling risks..."):
                contents = build_gemini_contents(images, extra_notes)
                report_text = call_gemini(settings["api_key"], settings["model"], contents)
        except RuntimeError as exc:
            st.error(f"Analysis failed: {exc}")
            return

        st.session_state["last_report"] = report_text

    # Display the most recent report, if any exists in this session.
    if "last_report" in st.session_state:
        st.markdown("---")
        st.subheader("📊 Engineering Review Report")

        sections = split_report_into_sections(st.session_state["last_report"])
        tab_titles = [t for t in REPORT_SECTIONS if t in sections] + \
                     [t for t in sections if t not in REPORT_SECTIONS]
        tabs = st.tabs([f"🔹 {t}" for t in tab_titles])
        for tab, title in zip(tabs, tab_titles):
            with tab:
                st.markdown(sections[title])

        st.download_button(
            "⬇️ Download full report (Markdown)",
            data=st.session_state["last_report"],
            file_name="pcb_review_report.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
