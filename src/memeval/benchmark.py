"""Four memory strategies, one recall test. Which one lets the agent still answer — and for how many
input tokens?

- **none** — the agent gets only the question. The floor: it can't know anything session-specific.
- **full** — the entire history is stuffed into the prompt. The ceiling on recall, but input tokens
  grow with the whole conversation, forever.
- **summary** — the history is compressed once into an LLM summary, and that summary is the context.
  Cheap and flat, but summarization drops specifics (a name, an exact number) it judged unimportant.
- **retrieval** — the history is chunked per turn, embedded, and only the turns most relevant to the
  question are put in the prompt. Cheap and flat like summary, but it can surface the *exact* turn.

Each strategy is scored on recall accuracy (did the answer contain the planted fact?) and mean input
tokens (the cost). The interesting comparison is retrieval vs full (does a few retrieved turns match
reading everything?) and retrieval vs summary (does keeping the raw turn beat compressing it?).
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from .data import QUESTIONS, SESSIONS, RecallQuestion
from .embed import top_k

SYSTEM = (
    "You are a helpful assistant with access to notes from earlier conversations with the user. "
    "Answer the user's question using ONLY those notes. Answer in as few words as possible. If the "
    "notes do not contain the answer, reply exactly \"I don't know\"."
)
SUMMARY_SYSTEM = (
    "Summarize the following conversation history into concise notes for future reference. Capture the "
    "important facts, decisions, names, and numbers. Keep it brief."
)


class FakeCompleter:
    """Offline agent for tests, modeling the one dynamic that matters: it can answer only from the
    context it was given. For a normal question it echoes back the provided notes (so a fact is
    'answered' iff it's actually in the context — exactly the real behavior). For a summarize request
    it returns a lossy summary that keeps a couple of headline facts but drops the specific
    numbers/names — mimicking how real summarization sheds detail. No network."""

    def complete(self, system: str, prompt: str):
        from .llm import Reply
        if system.startswith("Summarize"):
            summ = ("The user is building a project codenamed Halcyon, an internal analytics tool. "
                    "Manager is Dana. Frontend is Next.js. Rollout is phased.")  # drops region/numbers/color
            return Reply(summ, len(prompt.split()), len(summ.split()))
        if "Notes from earlier conversations:" in prompt:
            ctx = prompt.split("Notes from earlier conversations:", 1)[1].split("\n\nQuestion:", 1)[0]
            return Reply(ctx.strip(), len(prompt.split()), len(ctx.split()))
        return Reply("I don't know", len(prompt.split()), 3)


def _turns_with_tags() -> list[str]:
    """Every turn as an independently-retrievable chunk, tagged with its session."""
    return [f"(session {s.id}) {turn}" for s in SESSIONS for turn in s.turns]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def is_correct(output: str, q: RecallQuestion) -> bool:
    o = _norm(output)
    return any(_norm(a) in o for a in q.answers)


def _ask(client, context: str, question: str):
    prompt = (f"Notes from earlier conversations:\n{context}\n\nQuestion: {question}\nAnswer:"
              if context else f"Question: {question}\nAnswer:")
    t0 = time.perf_counter()
    reply = client.complete(SYSTEM, prompt)
    return reply, (time.perf_counter() - t0) * 1000


@dataclass
class StrategyResult:
    strategy: str
    accuracy: float
    mean_input_tokens: float
    mean_latency_ms: float
    setup_tokens: int = 0          # one-time cost (e.g. building the summary)
    n: int = 0
    details: list = field(default_factory=list)


def _summarize(client) -> tuple[str, int, int]:
    history = "\n".join(f"Session {s.id}: {s.text}" for s in SESSIONS)
    reply = client.complete(SUMMARY_SYSTEM, history)
    return reply.text, reply.input_tokens, reply.output_tokens


def run(client, embedder, k: int = 4, questions=QUESTIONS) -> dict:
    full_context = "\n".join(f"Session {s.id}: {s.text}" for s in SESSIONS)
    chunks = _turns_with_tags()
    summary_text, sum_in, sum_out = _summarize(client)

    strategies = {
        "none": lambda q: "",
        "full": lambda q: full_context,
        "summary": lambda q: summary_text,
        "retrieval": lambda q: "\n".join(chunks[i] for i in top_k(embedder, q.question, chunks, k)),
    }

    results = {}
    for name, ctx_fn in strategies.items():
        correct = 0
        tin = 0
        lat = 0.0
        details = []
        for q in questions:
            context = ctx_fn(q)
            reply, ms = _ask(client, context, q.question)
            ok = is_correct(reply.text, q)
            correct += int(ok)
            tin += reply.input_tokens
            lat += ms
            details.append({"qid": q.id, "correct": ok, "output": reply.text,
                            "input_tokens": reply.input_tokens})
        n = len(questions)
        results[name] = asdict(StrategyResult(
            strategy=name, accuracy=round(correct / n, 3),
            mean_input_tokens=round(tin / n, 1), mean_latency_ms=round(lat / n, 1),
            setup_tokens=(sum_in + sum_out) if name == "summary" else 0,
            n=n, details=details))
    return {"strategies": results, "k": k,
            "summary_preview": summary_text[:280]}


def format_report(r: dict) -> str:
    header = f"{'strategy':<11} {'accuracy':>9} {'mean_in_tok':>12} {'lat_ms':>8} {'setup_tok':>10}"
    lines = [header, "-" * len(header)]
    for name in ("none", "full", "summary", "retrieval"):
        s = r["strategies"][name]
        lines.append(f"{name:<11} {s['accuracy']:>9.3f} {s['mean_input_tokens']:>12.1f} "
                     f"{s['mean_latency_ms']:>8.1f} {s['setup_tokens']:>10}")
    return "\n".join(lines)
