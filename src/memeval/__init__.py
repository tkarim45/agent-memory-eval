"""agent-memory-eval — benchmark agent memory strategies (none / full-history / summary / vector
retrieval) on a multi-session recall test: which one still answers a question about something said
sessions ago, and at what token cost?"""
from .benchmark import format_report, run

__all__ = ["run", "format_report"]
