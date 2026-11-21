from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.llm import GroqChatClient, GroqClientError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optionally make one synthetic Groq API request."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send one real API request using local .env settings.",
    )
    args = parser.parse_args()

    if not args.live:
        print(
            "No API request sent. Review your current Groq access and model, "
            "then pass --live only if you want to make one request."
        )
        return 0

    try:
        client = GroqChatClient.from_env()
        response = client.chat(
            [
                {
                    "role": "system",
                    "content": "Reply with the exact text requested by the user.",
                },
                {
                    "role": "user",
                    "content": "Reply with exactly: GROQ_API_TEST_OK",
                },
            ],
            temperature=0,
            max_completion_tokens=32,
        )
    except GroqClientError as exc:
        # Print the safe client message, not any key or raw SDK response.
        print(f"Groq test failed: {exc}", file=sys.stderr)
        return 1

    print(f"Configured model: {response.model}")
    print(f"Response: {response.content}")
    print(
        "Token usage: "
        f"prompt={response.prompt_tokens}, "
        f"completion={response.completion_tokens}, "
        f"total={response.total_tokens}"
    )
    print(f"Request ID: {response.request_id or 'not provided'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())