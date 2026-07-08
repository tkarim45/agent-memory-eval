"""Offline tests — fake agent + hashing retriever, no keys, no network.

They verify the recall data is well-formed and that the strategy mechanics behave as designed with a
context-only fake agent: 'none' fails, 'full' recalls everything, retrieval recalls at far fewer
tokens than full, and summary loses the specific facts it dropped. Real accuracy numbers come from a
Bedrock run.
"""
from __future__ import annotations

from memeval.benchmark import FakeCompleter, is_correct, run
from memeval.data import QUESTIONS, SESSIONS
from memeval.embed import HashingEmbedder, top_k


def test_planted_facts_are_actually_in_their_sessions():
    for q in QUESTIONS:
        sess = next(s for s in SESSIONS if s.id == q.fact_session)
        assert any(a.lower() in sess.text.lower() for a in q.answers), q.id


def test_retrieval_finds_the_right_session_turn():
    chunks = [f"(session {s.id}) {t}" for s in SESSIONS for t in s.turns]
    q = next(q for q in QUESTIONS if q.id == "q9")   # accent color = teal, session 6
    idx = top_k(HashingEmbedder(), q.question, chunks, k=4)
    assert any("teal" in chunks[i].lower() for i in idx)


def test_none_fails_full_recalls():
    r = run(FakeCompleter(), HashingEmbedder(), k=4)
    s = r["strategies"]
    assert s["none"]["accuracy"] == 0.0            # no context -> can't answer
    assert s["full"]["accuracy"] == 1.0            # everything in context -> answers all


def test_retrieval_is_cheaper_than_full():
    r = run(FakeCompleter(), HashingEmbedder(), k=4)
    s = r["strategies"]
    assert s["retrieval"]["mean_input_tokens"] < s["full"]["mean_input_tokens"]
    assert s["retrieval"]["accuracy"] >= 0.5       # retrieves the right turns for most questions


def test_summary_loses_specific_facts():
    # the fake summary keeps Halcyon/Dana/Next.js but drops region/numbers/color
    r = run(FakeCompleter(), HashingEmbedder(), k=4)
    s = r["strategies"]["summary"]
    assert s["accuracy"] < r["strategies"]["full"]["accuracy"]   # compression drops detail
    assert s["setup_tokens"] > 0                                 # summarizing had a one-time cost


def test_is_correct_matching():
    q = next(q for q in QUESTIONS if q.id == "q2")   # us-west-2
    assert is_correct("We deployed to us-west-2.", q)
    assert not is_correct("I don't know", q)
