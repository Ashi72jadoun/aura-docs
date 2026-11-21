# Retrieval and Answer Evaluation

## Retrieval dataset

Use `data/evaluation/gold_questions.jsonl` for manually checked evaluation
questions. Each non-blank line must be one JSON object:

```json
{"id":"unique-id","language":"en","question":"A real question about an indexed PDF","relevant_sources":[{"document_name":"actual-file.pdf","page_number":3}]}