from types import SimpleNamespace
import unittest

from rag.conversation import ConversationSession, is_follow_up_question


class FakePipeline:
    """Offline stand-in: it does not call Groq."""

    def __init__(self):
        self.config = SimpleNamespace(max_question_chars=4000)
        self.questions = []

    def answer(self, question):
        self.questions.append(question)

        return SimpleNamespace(
            answer="- The library opens at nine. [S1]",
            cited_source_ids=["S1"],
            sources=[
                SimpleNamespace(
                    source_id="S1",
                    document_name="guide.pdf",
                    page_number=5,
                )
            ],
            llm_used=True,
            abstained=False,
            abstention_reason=None,
        )


class ConversationSessionTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = FakePipeline()
        self.session = ConversationSession(self.pipeline)

    def test_first_question_is_sent_without_history(self):
        question = "What are the library hours?"

        result = self.session.ask(question)

        self.assertEqual(self.pipeline.questions, [question])
        self.assertEqual(result.cited_source_ids, ["S1"])
        self.assertEqual(self.session.turn_count, 1)

    def test_follow_up_includes_recent_question_and_answer(self):
        self.session.ask("What are the library hours?")
        self.session.ask("What about Sundays?")

        sent_question = self.pipeline.questions[1]

        self.assertIn("Recent conversation (JSON)", sent_question)
        self.assertIn(
            '"user_question":"What are the library hours?"',
            sent_question,
        )
        self.assertIn(
            '"assistant_answer":"- The library opens at nine. [S1]"',
            sent_question,
        )
        self.assertIn(
            "Current user question:\nWhat about Sundays?",
            sent_question,
        )

    def test_unrelated_question_does_not_automatically_use_history(self):
        self.session.ask("What are the library hours?")
        unrelated_question = "What are the museum opening hours?"

        self.session.ask(unrelated_question)

        self.assertEqual(self.pipeline.questions[1], unrelated_question)

    def test_history_can_be_forced_for_a_question(self):
        self.session.ask("What are the library hours?")
        self.session.ask("What are the museum hours?", use_history=True)

        self.assertIn(
            "Recent conversation (JSON)",
            self.pipeline.questions[1],
        )

    def test_history_is_limited_to_max_turns(self):
        pipeline = FakePipeline()
        session = ConversationSession(pipeline, max_turns=2)

        session.ask("Tell me about the first subject.")
        session.ask("Tell me about the second subject.")
        session.ask("Tell me about the third subject.")

        self.assertEqual(session.turn_count, 2)
        self.assertEqual(
            [turn.question for turn in session.history],
            [
                "Tell me about the second subject.",
                "Tell me about the third subject.",
            ],
        )

    def test_clear_history_removes_saved_turns(self):
        self.session.ask("What are the library hours?")
        self.session.clear_history()

        self.assertEqual(self.session.history, ())
        self.assertEqual(self.session.turn_count, 0)

    def test_ui_history_contains_source_references(self):
        self.session.ask("What are the library hours?")

        messages = self.session.history_for_ui()

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(
            messages[1]["sources"][0],
            {
                "source_id": "S1",
                "document_name": "guide.pdf",
                "page_number": 5,
            },
        )

    def test_question_longer_than_limit_is_rejected(self):
        pipeline = FakePipeline()
        session = ConversationSession(
            pipeline,
            max_question_chars=10,
        )

        with self.assertRaises(ValueError):
            session.ask("This question is too long.")

    def test_follow_up_detector(self):
        self.assertTrue(is_follow_up_question("What about Sunday?"))
        self.assertTrue(is_follow_up_question("When does it close?"))
        self.assertFalse(is_follow_up_question("What are the library hours?"))


if __name__ == "__main__":
    unittest.main()