"""Create a fictional three-page PDF for a safe project demonstration."""

from __future__ import annotations

from pathlib import Path

import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "demo" / "demo_guide.pdf"


DEMO_PAGES = [
    (
        "Demo Guide - Opening Hours",
        (
            "Demo Library Opening Hours\n\n"
            "The demonstration library opens at 9:00 AM and closes at "
            "5:00 PM from Monday through Saturday.\n\n"
            "All facts in this guide are fictional and are included only "
            "to demonstrate the document question-answering application."
        ),
    ),
    (
        "Demo Guide - Closing Day",
        (
            "Demo Library Closing Day\n\n"
            "The demonstration library is closed every Sunday. It is open "
            "from Monday through Saturday.\n\n"
            "All facts in this guide are fictional and are included only "
            "to demonstrate the document question-answering application."
        ),
    ),
    (
        "Demo Guide - Membership",
        (
            "Demo Library Membership\n\n"
            "Membership at the demonstration library is free. A member can "
            "borrow up to five books for fourteen days.\n\n"
            "All facts in this guide are fictional and are included only "
            "to demonstrate the document question-answering application."
        ),
    ),
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    document = fitz.open()

    try:
        for page_number, (title, body) in enumerate(DEMO_PAGES, start=1):
            page = document.new_page(width=595, height=842)

            page.insert_text(
                (72, 80),
                title,
                fontname="helv",
                fontsize=19,
                color=(0.12, 0.25, 0.42),
            )

            page.insert_textbox(
                fitz.Rect(72, 115, 523, 720),
                body,
                fontname="helv",
                fontsize=13,
                lineheight=1.5,
                color=(0.12, 0.12, 0.12),
            )

            page.insert_text(
                (72, 790),
                f"Page {page_number} of {len(DEMO_PAGES)}",
                fontname="helv",
                fontsize=9,
                color=(0.4, 0.4, 0.4),
            )

        document.set_metadata(
            {
                "title": "Fictional Demo Library Guide",
                "author": "Document QA demo",
                "subject": "Synthetic content for software demonstration",
            }
        )
        document.save(str(OUTPUT_PATH))

    finally:
        document.close()

    print(f"Created synthetic demo PDF: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()