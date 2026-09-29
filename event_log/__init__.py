"""Event log schema and graph (Phase 2).

Named ``event_log`` instead of ``logging`` to avoid shadowing the Python stdlib.
"""

from event_log.schema import Event, hash_content, read_jsonl, write_jsonl

__all__ = ["Event", "hash_content", "read_jsonl", "write_jsonl"]
