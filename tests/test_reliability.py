from dataclasses import dataclass
import unittest

from rag.reliability import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    INSUFFICIENT_EVIDENCE_TOKEN,
    audit_citations,
    build_rag_messages,
    validate_rag_answer,
)


@dataclass
class FakeRAGAnswer:
    answer: str
    sources: list
    cited_source_ids: list
    llm_used: bool
    abstained: bool
    abstention_reason: str | None


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {
                "source_id": "S1",
                "chunk": "The library opens at nine.",
                "excerpt": "The library opens at nine.",
                "document_name": "guide.pdf",
                "page_number": 4,
            },
            {
                "source_id": "S2",
                "chunk": "The library closes at five.",
                "excerpt": "The library closes at five.",
                "document_name": "guide.pdf",
                "page_number": 5,
            },
        ]

    def test_valid_citations_pass(self):
        audit = audit_citations(
            "- The library opens at nine. [S1]",
            self.sources,
        )

        self.assertTrue(audit.is_valid)
        self.assertEqual(audit.valid_source_ids, ("S1",))
        self.assertEqual(audit.invalid_source_ids, ())

    def test_unknown_citation_fails(self):
        audit = audit_citations(
            "- The library opens at nine. [S99]",
            self.sources,
        )

        self.assertFalse(audit.is_valid)
        self.assertEqual(audit.invalid_source_ids, ("S99",))

    def test_every_nonempty_line_needs_a_valid_citation(self):
        audit = audit_citations(
            "- The library opens at nine. [S1]\n"
            "- This second line has no citation.",
            self.sources,
        )

        self.assertFalse(audit.is_valid)
        self.assertEqual(audit.uncited_lines, (2,))

    def test_citation_without_an_answer_is_not_valid(self):
        audit = audit_citations("[S1]", self.sources)

        self.assertFalse(audit.is_valid)
        self.assertFalse(audit.has_claims)

    def test_prompt_contains_source_metadata(self):
        messages = build_rag_messages(
            "When does the library open?",
            self.sources,
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("untrusted data", messages[0]["content"].lower())
        self.assertIn("guide.pdf", messages[1]["content"])
        self.assertIn('"page_number": 4', messages[1]["content"])
        self.assertIn('"source_id": "S1"', messages[1]["content"])

    def test_valid_answer_keeps_only_valid_cited_ids(self):
        result = FakeRAGAnswer(
            answer="- The library opens at nine. [S1]",
            sources=self.sources,
            cited_source_ids=[],
            llm_used=True,
            abstained=False,
            abstention_reason=None,
        )

        checked = validate_rag_answer(result)

        self.assertFalse(checked.abstained)
        self.assertEqual(checked.cited_source_ids, ["S1"])

    def test_answer_with_invalid_citation_is_withheld(self):
        result = FakeRAGAnswer(
            answer="- The library opens at nine. [S99]",
            sources=self.sources,
            cited_source_ids=[],
            llm_used=True,
            abstained=False,
            abstention_reason=None,
        )

        checked = validate_rag_answer(result)

        self.assertTrue(checked.abstained)
        self.assertEqual(checked.cited_source_ids, [])
        self.assertEqual(checked.abstention_reason, "unknown_citation_id")

    def test_insufficient_evidence_token_becomes_safe_abstention(self):
        result = FakeRAGAnswer(
            answer=INSUFFICIENT_EVIDENCE_TOKEN,
            sources=self.sources,
            cited_source_ids=[],
            llm_used=True,
            abstained=False,
            abstention_reason=None,
        )

        checked = validate_rag_answer(result)

        self.assertTrue(checked.abstained)
        self.assertEqual(checked.answer, INSUFFICIENT_EVIDENCE_MESSAGE)
        self.assertEqual(
            checked.abstention_reason,
            "model_reported_insufficient_evidence",
        )


if __name__ == "__main__":
    unittest.main()