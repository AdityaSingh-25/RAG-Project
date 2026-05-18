from langchain_ollama import ChatOllama

from rag_engine.config.settings import Settings
from rag_engine.utils.tokenization import count_tokens


class ModelRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def for_question(self, question: str, context: str) -> ChatOllama:
        model = self._choose_model(question, context)
        return ChatOllama(
            base_url=self.settings.ollama_base_url,
            model=model,
            temperature=0.1,
        )

    def _choose_model(self, question: str, context: str) -> str:
        question_tokens = count_tokens(question, encoding_name=self.settings.token_encoding)
        context_tokens = count_tokens(context, encoding_name=self.settings.token_encoding)
        complex_question = self._is_complex_question(question, question_tokens)
        high_context = context_tokens > self.settings.max_context_tokens * 0.6
        if complex_question or high_context or question_tokens > 90:
            return self.settings.chat_model_reasoning
        return self.settings.chat_model_fast

    def _is_complex_question(self, question: str, question_tokens: int) -> bool:
        normalized = question.lower()
        markers = (
            "compare",
            "reason",
            "why",
            "tradeoff",
            "derive",
            "explain",
            "evaluate",
            "best",
            "recommend",
        )
        score = sum(1 for marker in markers if marker in normalized)
        return score >= 2 or question_tokens > 80

