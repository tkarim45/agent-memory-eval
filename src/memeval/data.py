"""A synthetic multi-session history with planted facts, plus recall questions that ask about them.

An agent that talks to a user over many sessions accumulates a history far bigger than one context
window. The question this benchmark answers: *which memory strategy lets the agent still answer a
question about something said sessions ago — and at what token cost?* So the data is a sequence of
short "sessions" (each a few user turns), most of it plausible chatter, with specific facts planted in
particular sessions (a name, a number, a preference, a decision). The recall questions each target one
planted fact and have a short, checkable gold answer. The facts are spread across sessions and
surrounded by distractor turns, so a strategy only scores if it actually surfaces the *right* session,
not just a lot of context.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Session:
    id: int
    turns: tuple[str, ...]      # user messages in this session

    @property
    def text(self) -> str:
        return " ".join(self.turns)


@dataclass(frozen=True)
class RecallQuestion:
    id: str
    question: str
    answers: tuple[str, ...]    # acceptable substrings (case-insensitive)
    fact_session: int           # which session the answer was planted in (for analysis)


SESSIONS: tuple[Session, ...] = (
    Session(0, (
        "Hey, I'm setting up a new project and wanted to think out loud with you.",
        "It's an internal analytics tool for the ops team.",
        "By the way, the project is codenamed Halcyon.",
        "I usually work best in the mornings, so I'll probably ping you early.",
    )),
    Session(1, (
        "Quick update: we picked our cloud region today.",
        "We're deploying Halcyon to us-west-2 because most of our users are on the west coast.",
        "Also the team is small, just four engineers for now.",
    )),
    Session(2, (
        "I had a long meeting about the data model.",
        "We decided the primary datastore will be Postgres, not DynamoDB, mostly for the joins.",
        "My manager Dana wants a demo by the end of the quarter.",
        "Lunch was terrible today, unrelated.",
    )),
    Session(3, (
        "Spent the morning on auth.",
        "We're going with short-lived tokens that rotate every 30 minutes for the admin API.",
        "The frontend is Next.js, which the team already knows well.",
    )),
    Session(4, (
        "Random thought: I want the dashboard to load in under 2 seconds on the p95.",
        "That's the performance bar I'm holding us to.",
        "We also added a new engineer, so we're five now.",
    )),
    Session(5, (
        "Talked to security today.",
        "They require that all audit logs be retained for 400 days, which surprised me.",
        "Otherwise a calm day.",
    )),
    Session(6, (
        "Design review went well.",
        "We agreed the color accent for Halcyon's UI will be teal.",
        "Dana liked it. Feeling good about the demo.",
    )),
    Session(7, (
        "Planning the rollout.",
        "We'll launch to the ops team first, then finance, then the whole company in three waves.",
        "I think that's the safest sequencing.",
    )),
)

QUESTIONS: tuple[RecallQuestion, ...] = (
    RecallQuestion("q1", "What is the project's codename?", ("Halcyon",), 0),
    RecallQuestion("q2", "Which cloud region are we deploying to?", ("us-west-2",), 1),
    RecallQuestion("q3", "Which primary datastore did we choose?", ("Postgres",), 2),
    RecallQuestion("q4", "Who is my manager?", ("Dana",), 2),
    RecallQuestion("q5", "How often do the admin API tokens rotate?", ("30 minutes", "30 min"), 3),
    RecallQuestion("q6", "What frontend framework is the project using?", ("Next.js", "Next"), 3),
    RecallQuestion("q7", "What is the p95 dashboard load-time target?", ("2 second", "2 sec", "under 2"), 4),
    RecallQuestion("q8", "How long must audit logs be retained?", ("400 day", "400"), 5),
    RecallQuestion("q9", "What accent color did we pick for the UI?", ("teal",), 6),
    RecallQuestion("q10", "Which team gets the rollout first?", ("ops",), 7),
)


def sessions_before(_: RecallQuestion) -> list[Session]:
    """All sessions are 'past history' relative to the recall questions (asked at the end)."""
    return list(SESSIONS)
