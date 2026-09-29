"""LLM package for Mode A decision backends + embeddings."""

from llm.decision import (
    AgentDecision,
    DecisionPoint,
    InboxMessage,
    MockDecisionBackend,
    OpenAIDecisionBackend,
    make_backend,
)
from llm.embeddings import enrich_event_embeddings
