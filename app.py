"""AURA DoCS: multilingual PDF chat with text and optional voice input."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
UI_COMPONENT_PATH = PROJECT_ROOT / "ui_components" / "aura_controls"

load_dotenv(PROJECT_ROOT / ".env", override=False)

from rag.app_factory import create_pipeline  # noqa: E402
from rag.conversation import ConversationSession  # noqa: E402


st.set_page_config(
    page_title="AURA DoCS",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

_AURA_CONTROLS = components.declare_component(
    "aura_doc_controls",
    path=str(UI_COMPONENT_PATH),
)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _api_key_is_configured() -> bool:
    value = os.getenv("GROQ_API_KEY", "").strip().casefold()
    if not value:
        return False

    return not any(
        marker in value
        for marker in (
            "replace_with",
            "your_actual",
            "paste_your",
        )
    )


@st.cache_resource(show_spinner=False)
def _get_pipeline():
    return create_pipeline()


def _add_styles(dark_mode: bool) -> None:
    if dark_mode:
        palette = {
            "bg": "#1c1b25",
            "text": "#f8f1f4",
            "muted": "#c1b7c2",
            "card": "#282631",
            "sidebar": "#22212b",
            "border": "#45414f",
            "coral": "#ff8968",
            "blue": "#929ff8",
            "shadow": "rgba(0, 0, 0, 0.25)",
        }
    else:
        palette = {
            "bg": "#fff7f1",
            "text": "#342a32",
            "muted": "#786f78",
            "card": "#fffefd",
            "sidebar": "#fffaf7",
            "border": "#f1d9ce",
            "coral": "#ff805d",
            "blue": "#8998f3",
            "shadow": "rgba(105, 69, 53, 0.10)",
        }

    css = """
    <style>
    :root {
        --aura-bg: __BG__;
        --aura-text: __TEXT__;
        --aura-muted: __MUTED__;
        --aura-card: __CARD__;
        --aura-sidebar: __SIDEBAR__;
        --aura-border: __BORDER__;
        --aura-coral: __CORAL__;
        --aura-blue: __BLUE__;
        --aura-shadow: __SHADOW__;
    }

    .stApp {
        color: var(--aura-text);
        background:
            radial-gradient(
                circle at 12% 3%,
                rgba(255, 128, 93, 0.13),
                transparent 30rem
            ),
            radial-gradient(
                circle at 92% 12%,
                rgba(137, 152, 243, 0.14),
                transparent 27rem
            ),
            var(--aura-bg);
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
    }

    section[data-testid="stSidebar"] {
        background: var(--aura-sidebar);
        border-right: 1px solid var(--aura-border);
    }

    section[data-testid="stSidebar"] > div {
        background: transparent;
    }

    h1, h2, h3, p, label {
        color: var(--aura-text);
    }

    .aura-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        min-height: 5.2rem;
    }

    .aura-logo {
        position: relative;
        flex: 0 0 56px;
        width: 56px;
        height: 52px;
    }

    .aura-logo span {
        position: absolute;
        width: 30px;
        height: 36px;
        border: 2px solid rgba(255, 255, 255, 0.9);
        border-radius: 7px;
        box-shadow: 0 5px 12px var(--aura-shadow);
    }

    .aura-logo span:nth-child(1) {
        top: 1px;
        left: 3px;
        background: #9bdc72;
        transform: rotate(-7deg);
    }

    .aura-logo span:nth-child(2) {
        top: 6px;
        left: 15px;
        background: #ff7f9b;
        transform: rotate(3deg);
    }

    .aura-logo span:nth-child(3) {
        top: 11px;
        left: 26px;
        background: #55c8e9;
        transform: rotate(8deg);
    }

    .aura-eyebrow {
        color: var(--aura-coral);
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .aura-heading {
        margin: 0.1rem 0 0;
        color: var(--aura-text);
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 800;
        letter-spacing: -0.045em;
        line-height: 1.08;
    }

    .aura-heading span {
        color: var(--aura-coral);
    }

    .aura-tagline {
        margin-top: 0.3rem;
        color: var(--aura-muted);
        font-size: 0.95rem;
    }

    .aura-side-brand {
        padding-bottom: 0.6rem;
        color: var(--aura-text);
        font-size: 1.2rem;
        font-weight: 800;
    }

    .aura-side-brand span {
        color: var(--aura-coral);
    }

    .aura-card {
        padding: 1.4rem;
        border: 1px solid var(--aura-border);
        border-radius: 22px;
        background: var(--aura-card);
        box-shadow: 0 14px 35px var(--aura-shadow);
    }

    .aura-welcome-title {
        margin: 0.2rem 0 0.5rem;
        color: var(--aura-text);
        font-size: 1.6rem;
        font-weight: 750;
    }

    .aura-welcome-copy {
        margin: 0;
        color: var(--aura-muted);
        line-height: 1.7;
    }

    .aura-pill {
        display: inline-block;
        margin: 0.75rem 0.35rem 0 0;
        padding: 0.36rem 0.7rem;
        border: 1px solid var(--aura-border);
        border-radius: 999px;
        background: var(--aura-card);
        color: var(--aura-muted);
        font-size: 0.82rem;
    }

    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        min-height: 2.7rem;
        border: 1px solid var(--aura-border);
        border-radius: 14px;
        color: var(--aura-text);
        background: var(--aura-card);
    }

    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
        border: 0;
        color: #ffffff;
        background: linear-gradient(
            110deg,
            var(--aura-coral),
            var(--aura-blue)
        );
    }

    div[data-baseweb="textarea"] > div {
        border-color: var(--aura-border);
        border-radius: 14px;
        background: var(--aura-card);
    }

    div[data-baseweb="textarea"] textarea {
        color: var(--aura-text);
    }

    div[data-testid="stChatMessage"] {
        margin: 0.6rem 0;
        padding: 0.55rem 0.75rem;
        border: 1px solid var(--aura-border);
        border-radius: 18px;
        background: var(--aura-card);
        box-shadow: 0 6px 18px var(--aura-shadow);
    }

    div[data-testid="stChatMessage"] p {
        color: var(--aura-text);
    }

    [data-testid="stExpander"] {
        border-color: var(--aura-border);
        border-radius: 14px;
        background: var(--aura-card);
    }

    .aura-footer-spacer {
        height: 2.5rem;
    }

    .aura-footer {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.45rem 0.8rem;
        padding: 1rem 0 0.45rem;
        border-top: 1px solid var(--aura-border);
        color: var(--aura-muted);
        font-size: 0.82rem;
        text-align: center;
    }

    .aura-footer strong {
        color: var(--aura-coral);
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 0.7rem;
        }

        .aura-card {
            padding: 1rem;
        }
    }
    </style>
    """

    for placeholder, value in {
        "__BG__": palette["bg"],
        "__TEXT__": palette["text"],
        "__MUTED__": palette["muted"],
        "__CARD__": palette["card"],
        "__SIDEBAR__": palette["sidebar"],
        "__BORDER__": palette["border"],
        "__CORAL__": palette["coral"],
        "__BLUE__": palette["blue"],
        "__SHADOW__": palette["shadow"],
    }.items():
        css = css.replace(placeholder, value)

    st.markdown(css, unsafe_allow_html=True)


def _consume_component_event(
    value: Any,
    session_key: str,
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None

    event_id = str(value.get("event_id", "")).strip()
    if not event_id:
        return None

    last_event_key = f"{session_key}_last_event_id"
    if st.session_state.get(last_event_key) == event_id:
        return None

    st.session_state[last_event_key] = event_id
    return value


def _render_header(dark_mode: bool) -> Mapping[str, Any] | None:
    title_column, mode_column = st.columns(
        [0.72, 0.28],
        vertical_alignment="center",
    )

    with title_column:
        st.markdown(
            """
            <div class="aura-header">
                <div class="aura-logo" aria-hidden="true">
                    <span></span><span></span><span></span>
                </div>
                <div>
                    <div class="aura-eyebrow">A thoughtful way to explore</div>
                    <h1 class="aura-heading">AURA <span>DoCS</span></h1>
                    <div class="aura-tagline">
                        Your documents, ready for a conversation.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with mode_column:
        value = _AURA_CONTROLS(
            mode="theme",
            dark_mode=dark_mode,
            enabled=True,
            placeholder="",
            key="aura_theme_control",
            default=None,
        )

    return _consume_component_event(value, "aura_theme")


def _indexed_document_names(pipeline: Any) -> list[str]:
    method = getattr(pipeline.vector_store, "list_document_names", None)
    if not callable(method):
        return []

    try:
        result = method() or []
    except Exception:
        return []

    if isinstance(result, Mapping):
        result = result.get(
            "document_names",
            result.get("names", []),
        )

    if isinstance(result, (str, bytes)):
        return [str(result)]

    try:
        items = list(result)
    except TypeError:
        return []

    names: list[str] = []

    for item in items:
        if isinstance(item, str):
            names.append(item)
        else:
            name = _get(
                item,
                "document_name",
                "filename",
                "name",
                default=None,
            )
            if name:
                names.append(str(name))

    return names


def _render_sources(sources: Any, show_excerpts: bool) -> None:
    if not sources:
        return

    try:
        source_list = list(sources)
    except TypeError:
        return

    if not source_list:
        return

    with st.expander(f"Sources · {len(source_list)}", expanded=False):
        for source in source_list:
            source_id = str(_get(source, "source_id", "S?"))
            document_name = str(
                _get(source, "document_name", "Unknown PDF")
            )
            page_number = _get(source, "page_number", None)

            page_text = (
                f"page {page_number}"
                if page_number is not None
                else "page unavailable"
            )

            st.markdown(
                f"**[{source_id}]** · `{document_name}` · {page_text}"
            )

            if show_excerpts:
                excerpt = _get(source, "excerpt", None)
                if not excerpt:
                    excerpt = _get(source, "chunk", None)
                if excerpt:
                    st.text(str(excerpt))

            st.divider()


def _render_saved_history(conversation: ConversationSession) -> None:
    for turn in conversation.history:
        with st.chat_message("user", avatar="🙂"):
            st.markdown(turn.question)

        with st.chat_message("assistant", avatar="✨"):
            st.markdown(turn.answer)
            _render_sources(turn.sources, show_excerpts=False)


def _queue_voice_question() -> None:
    question = str(
        st.session_state.get("aura_voice_draft", "")
    ).strip()

    if question:
        st.session_state["aura_pending_question"] = question

    st.session_state["aura_voice_draft"] = ""


def _close_voice_panel() -> None:
    st.session_state["aura_voice_open"] = False


def _transcribe_audio(audio_file: Any) -> str:
    if not _api_key_is_configured():
        raise RuntimeError(
            "Voice transcription is not configured in the root .env file."
        )

    audio_bytes = audio_file.getvalue()
    if not audio_bytes:
        raise ValueError("The recording is empty.")

    if len(audio_bytes) > 20 * 1024 * 1024:
        raise ValueError("Please record a shorter question.")

    from groq import Groq

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY", "").strip(),
        max_retries=0,
        timeout=45.0,
    )

    model_name = os.getenv(
        "GROQ_TRANSCRIPTION_MODEL",
        "whisper-large-v3-turbo",
    ).strip()

    response = client.audio.transcriptions.create(
        file=(
            getattr(audio_file, "name", None) or "aura-question.wav",
            audio_bytes,
        ),
        model=model_name,
        response_format="json",
    )

    if isinstance(response, Mapping):
        transcript = response.get("text", "")
    else:
        transcript = getattr(response, "text", "")

    return str(transcript or "").strip()


def _render_voice_panel(has_documents: bool) -> None:
    if not st.session_state.get("aura_voice_open", False):
        return

    with st.container(border=True):
        title_column, close_column = st.columns(
            [0.82, 0.18],
            vertical_alignment="center",
        )

        with title_column:
            st.markdown("#### Ask by voice")

        with close_column:
            st.button(
                "Close",
                use_container_width=True,
                on_click=_close_voice_panel,
                key="aura_close_voice_panel",
            )

        st.caption("Record a question, review the words, then send it.")
        st.caption(
            "Audio is sent for transcription only after you select "
            "Transcribe and give consent."
        )

        audio_input = getattr(st, "audio_input", None)
        if not callable(audio_input):
            st.error(
                "Microphone recording needs a newer Streamlit version. "
                "Update Streamlit to version 1.40 or newer."
            )
            return

        audio_file = audio_input(
            "Record your question",
            sample_rate=16000,
            key="aura_audio_input",
        )

        consent = st.checkbox(
            "Send this recording for transcription",
            key="aura_voice_consent",
        )

        transcribe_clicked = st.button(
            "Transcribe",
            type="primary",
            use_container_width=True,
            disabled=(
                audio_file is None
                or not consent
                or not _api_key_is_configured()
            ),
            key="aura_transcribe_button",
        )

        if transcribe_clicked and audio_file is not None:
            try:
                with st.spinner("Transcribing…"):
                    transcript = _transcribe_audio(audio_file)

                if transcript:
                    st.session_state["aura_voice_draft"] = transcript
                    st.success("Transcript ready. Review it below.")
                else:
                    st.warning("No speech was detected. Try again.")

            except Exception as exc:
                st.error(
                    "Transcription failed. Check your connection and "
                    "account/model availability."
                )
                st.caption(f"Error type: {type(exc).__name__}")

        draft = str(
            st.session_state.get("aura_voice_draft", "")
        ).strip()

        if draft:
            st.text_area(
                "Review or edit your question",
                key="aura_voice_draft",
                height=100,
            )

            current_draft = str(
                st.session_state.get("aura_voice_draft", "")
            ).strip()

            st.button(
                "Ask this question",
                type="primary",
                use_container_width=True,
                disabled=not has_documents or not current_draft,
                on_click=_queue_voice_question,
                key="aura_send_voice_question",
            )


def _render_composer(
    has_documents: bool,
    dark_mode: bool,
) -> Mapping[str, Any] | None:
    value = _AURA_CONTROLS(
        mode="composer",
        dark_mode=dark_mode,
        enabled=has_documents,
        placeholder="Ask about your PDFs in any language…",
        key="aura_composer_control",
        default=None,
    )

    return _consume_component_event(value, "aura_composer")


def _render_sidebar(
    pipeline: Any,
    conversation: ConversationSession,
) -> None:
    with st.sidebar:
        st.markdown(
            '<div class="aura-side-brand">AURA <span>DoCS</span></div>',
            unsafe_allow_html=True,
        )

        st.subheader("Your library")
        st.caption("Add a PDF, then explore its contents.")

        uploaded_pdf = st.file_uploader(
            "Choose a text-based PDF",
            type=["pdf"],
            accept_multiple_files=False,
            help="Scanned/image-only PDFs need OCR, which is not included.",
        )

        replace_existing = st.checkbox(
            "Replace a PDF with the same filename",
            value=True,
        )

        if st.button(
            "＋ Add PDF to library",
            type="primary",
            use_container_width=True,
            disabled=uploaded_pdf is None,
        ) and uploaded_pdf is not None:
            try:
                with st.spinner("Reading and indexing your PDF…"):
                    result = pipeline.index_pdf(
                        pdf_bytes=uploaded_pdf.getvalue(),
                        filename=uploaded_pdf.name,
                        replace_existing=replace_existing,
                    )

                st.success(
                    f"Added **{result['document_name']}** — "
                    f"{result['pages_indexed']} pages."
                )

            except Exception as exc:
                st.error(f"PDF indexing failed: {exc}")

        st.divider()
        st.markdown("**Indexed PDFs**")

        names = _indexed_document_names(pipeline)
        if names:
            for name in names:
                st.markdown(f"📄 `{name}`")
        else:
            st.caption("Your library is empty for now.")

        with st.expander("Privacy", expanded=False):
            st.caption(
                "For answers, your question and matching PDF passages are "
                "sent to the configured answer service. Voice audio is sent "
                "separately for transcription only after you choose Transcribe."
            )

        if st.button("Clear this chat", use_container_width=True):
            conversation.clear_history()
            st.rerun()


def _answer_question(
    question: str,
    conversation: ConversationSession,
) -> None:
    with st.chat_message("user", avatar="🙂"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="✨"):
        try:
            with st.spinner("Looking through your documents…"):
                response = conversation.ask(question)

            st.markdown(response.answer)

            if response.abstained:
                st.caption(
                    "I couldn’t verify an answer from the retrieved passages."
                )

            _render_sources(
                response.sources,
                show_excerpts=True,
            )

        except Exception as exc:
            st.error("I couldn’t complete that search. Please try again.")
            st.caption(f"Error type: {type(exc).__name__}")


def _render_footer() -> None:
    st.markdown(
        """
        <div class="aura-footer-spacer"></div>
        <footer class="aura-footer">
            <strong>AURA DoCS</strong>
            <span>Engineered with Python, Groq, SQLite & by AJ:)</span>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def _main() -> None:
    st.session_state.setdefault("aura_dark_mode", False)
    st.session_state.setdefault("aura_voice_open", False)
    st.session_state.setdefault("aura_voice_draft", "")

    dark_mode = bool(st.session_state.get("aura_dark_mode", False))
    _add_styles(dark_mode)

    theme_event = _render_header(dark_mode)
    if theme_event and theme_event.get("kind") == "theme":
        next_dark_mode = bool(theme_event.get("dark_mode", False))

        if next_dark_mode != dark_mode:
            st.session_state["aura_dark_mode"] = next_dark_mode
            st.rerun()

    try:
        with st.spinner("Getting your workspace ready…"):
            pipeline = _get_pipeline()

    except Exception as exc:
        st.error(
            "AURA DoCS couldn’t finish setting up. Check the local "
            "dependencies, model and SQLite store."
        )
        st.caption(f"Error type: {type(exc).__name__}")
        st.stop()

    if "aura_conversation" not in st.session_state:
        st.session_state["aura_conversation"] = ConversationSession(
            pipeline=pipeline,
            max_turns=20,
            context_turns=3,
            max_history_chars=3000,
        )

    conversation: ConversationSession = st.session_state[
        "aura_conversation"
    ]

    _render_sidebar(pipeline, conversation)
    indexed_names = _indexed_document_names(pipeline)

    if not conversation.history:
        st.markdown(
            """
            <div class="aura-card">
                <div class="aura-eyebrow">A fresh page</div>
                <div class="aura-welcome-title">
                    Let’s find something useful.
                </div>
                <p class="aura-welcome-copy">
                    Add a PDF, then ask in the language that feels natural.
                </p>
                <span class="aura-pill">Explore your PDFs</span>
                <span class="aura-pill">Ask follow-up questions</span>
                <span class="aura-pill">See page references</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not indexed_names:
            st.info("Add a PDF in **Your library** to begin.")

    _render_saved_history(conversation)

    pending_question = st.session_state.pop(
        "aura_pending_question",
        None,
    )
    if pending_question:
        _answer_question(str(pending_question), conversation)

    st.markdown("#### Continue the conversation")

    # Show the recorder panel above the composer so it isn't below the screen.
    if st.session_state.get("aura_voice_open", False):
        _render_voice_panel(has_documents=bool(indexed_names))

    composer_event = _render_composer(
        has_documents=bool(indexed_names),
        dark_mode=bool(st.session_state.get("aura_dark_mode", False)),
    )

    if composer_event:
        event_kind = composer_event.get("kind")

        if event_kind == "submit":
            question = str(composer_event.get("question", "")).strip()

            if question and indexed_names:
                st.session_state["aura_pending_question"] = question
                st.rerun()

        elif event_kind == "voice":
            st.session_state["aura_voice_open"] = True
            st.rerun()

    _render_footer()

if __name__ == "__main__":
    _main()