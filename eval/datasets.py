"""Dataset loaders.

`synthetic()` — tiny offline scenario. `locomo()` — real LoCoMo. `multiday()` — evolving facts +
staleness detector. `conflicts()` — adversarial near-duplicates + temporal conflicts. `longmemeval()`
— LongMemEval (HuggingFace `xiaowu0162/longmemeval-cleaned`). All return the same Event/Question/Sample
shape so runner, metrics, and retention harnesses are unchanged.
"""
from __future__ import annotations

import datetime as _dt
import json
import random
import re
from pathlib import Path

from .schema import Dataset, Event, Question, Sample

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_SESSION_RE = re.compile(r"^session_(\d+)$")

# LongMemEval timestamps look like '2023/05/20 (Sat) 02:21'. Parse to epoch seconds so memories
# carry real event time (the benchmark's temporal-reasoning / knowledge-update signal).
_LME_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})\D+(\d{2}):(\d{2})")


def _parse_lme_date(value: str | None) -> float | None:
    if not value:
        return None
    m = _LME_DATE_RE.search(value)
    if not m:
        return None
    y, mo, d, h, mi = (int(x) for x in m.groups())
    try:
        return _dt.datetime(y, mo, d, h, mi, tzinfo=_dt.timezone.utc).timestamp()
    except ValueError:
        return None


def synthetic() -> Dataset:
    """An evolving project where a handful of durable facts/decisions are interleaved
    with disposable chatter, then queried later. Distractors + a tight token budget
    make recall non-trivial — the 'needle survives the noise' property real
    long-horizon memory needs."""
    events = [
        Event("e1", "The project codename is Polaris.", kind="fact", importance=4),
        Event("e2", "We chatted about the weekend and the weather for a bit.", kind="chat"),
        Event("e3", "Decision: the primary database is PostgreSQL, not MySQL.", kind="constraint", importance=5),
        Event("e4", "Aside: there's a great coffee place that just opened downtown.", kind="chat"),
        Event("e5", "The launch date was moved to September 14.", kind="fact", importance=5),
        Event("e6", "The user prefers all answers in metric units.", kind="preference", importance=3),
        Event("e7", "Constraint: never store user PII in plaintext, anywhere.", kind="constraint", importance=5),
        Event("e8", "We talked about favorite movies and a show everyone is watching.", kind="chat"),
        Event("e9", "Decision: API responses must stay under 200ms at p95.", kind="constraint", importance=4),
        Event("e10", "The monthly infrastructure budget ceiling is 2000 euros.", kind="fact", importance=4),
        Event("e11", "Small talk about a vacation to the coast next month.", kind="chat"),
        Event("e12", "The team stand-up moved to 10am on Tuesdays.", kind="note", importance=2),
        Event("e13", "Reminder to buy more coffee for the office kitchen.", kind="chat"),
        Event("e14", "The lead backend engineer on Polaris is Mara.", kind="fact", importance=3),
    ]
    questions = [
        Question("q1", "Which database did we decide to use?", "PostgreSQL", ["e3"]),
        Question("q2", "When is the launch happening?", "September 14", ["e5"]),
        Question("q3", "What is the monthly infrastructure budget?", "2000 euros", ["e10"]),
        Question("q4", "Are there any rules about handling user data?", "never store user PII in plaintext", ["e7"]),
        Question("q5", "What is the project codename?", "Polaris", ["e1"]),
        Question("q6", "Who is the lead backend engineer?", "Mara", ["e14"]),
    ]
    return Dataset(name="synthetic-v0", samples=[Sample("s1", events, questions)])


def synthetic_es() -> Dataset:
    """Spanish mirror of `synthetic()` (same ids, same gold structure): measures whether the
    configured embedder actually retrieves by meaning in a non-English language — an
    English-only embedder (bge-base) degrades here; a multilingual one should not."""
    events = [
        Event("e1", "El nombre en clave del proyecto es Polaris.", kind="fact", importance=4),
        Event("e2", "Charlamos un rato sobre el fin de semana y el tiempo.", kind="chat"),
        Event("e3", "Decisión: la base de datos principal es PostgreSQL, no MySQL.", kind="constraint", importance=5),
        Event("e4", "Por cierto: acaban de abrir una cafetería buenísima en el centro.", kind="chat"),
        Event("e5", "La fecha de lanzamiento se movió al 14 de septiembre.", kind="fact", importance=5),
        Event("e6", "El usuario prefiere todas las respuestas en unidades métricas.", kind="preference", importance=3),
        Event("e7", "Restricción: nunca almacenar PII de usuarios en texto plano, en ningún sitio.", kind="constraint", importance=5),
        Event("e8", "Hablamos de películas favoritas y de una serie que todos están viendo.", kind="chat"),
        Event("e9", "Decisión: las respuestas de la API deben mantenerse por debajo de 200ms en p95.", kind="constraint", importance=4),
        Event("e10", "El techo de presupuesto mensual de infraestructura es de 2000 euros.", kind="fact", importance=4),
        Event("e11", "Conversación trivial sobre unas vacaciones en la costa el mes que viene.", kind="chat"),
        Event("e12", "La reunión diaria del equipo se movió a las 10 de la mañana los martes.", kind="note", importance=2),
        Event("e13", "Recordatorio de comprar más café para la cocina de la oficina.", kind="chat"),
        Event("e14", "La ingeniera backend principal de Polaris es Mara.", kind="fact", importance=3),
    ]
    questions = [
        Question("q1", "¿Qué base de datos decidimos usar?", "PostgreSQL", ["e3"]),
        Question("q2", "¿Cuándo es el lanzamiento?", "14 de septiembre", ["e5"]),
        Question("q3", "¿Cuál es el presupuesto mensual de infraestructura?", "2000 euros", ["e10"]),
        Question("q4", "¿Hay alguna regla sobre el manejo de datos de usuarios?", "nunca almacenar PII de usuarios en texto plano", ["e7"]),
        Question("q5", "¿Cuál es el nombre en clave del proyecto?", "Polaris", ["e1"]),
        Question("q6", "¿Quién es la ingeniera backend principal?", "Mara", ["e14"]),
    ]
    return Dataset(name="synthetic-es-v0", samples=[Sample("s1", events, questions)])


def locomo(
    path: str | Path | None = None,
    *,
    max_conversations: int | None = None,
    drop_adversarial: bool = True,
) -> Dataset:
    """Load LoCoMo (snap-research/locomo).

    Each conversation -> one Sample: dialog turns become Event[] (id = dia_id like
    'D1:3'); QA pairs become Question[] with gold_event_ids = the QA's `evidence`
    turn ids. Adversarial questions (category 5, unanswerable) are dropped by default.
    """
    path = Path(path) if path else _DATA_DIR / "locomo10.json"
    if not path.exists():
        raise FileNotFoundError(
            f"LoCoMo not found at {path}. Download data/locomo10.json from "
            "https://github.com/snap-research/locomo (see README 'Next steps')."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if max_conversations:
        raw = raw[:max_conversations]

    samples: list[Sample] = []
    for idx, conv in enumerate(raw):
        convo = conv["conversation"]
        session_keys = sorted(
            (k for k in convo if _SESSION_RE.match(k)),
            key=lambda k: int(_SESSION_RE.match(k).group(1)),
        )
        events: list[Event] = []
        for skey in session_keys:
            for turn in convo[skey]:
                text = turn.get("text")
                if not text:
                    continue
                events.append(
                    Event(
                        id=turn["dia_id"],
                        content=f"{turn.get('speaker', '?')}: {text}",
                        kind="chat",
                        speaker=turn.get("speaker"),
                        metadata={"session": skey},
                    )
                )

        questions: list[Question] = []
        for qi, qa in enumerate(conv.get("qa", [])):
            if drop_adversarial and qa.get("category") == 5:
                continue
            ans = qa.get("answer")
            evidence = qa.get("evidence") or []
            category_num = qa.get("category", 1)
            # Map LoCoMo numeric categories to descriptive names
            # Based on LoCoMo paper: categories 1-4 are answerable, 5 is adversarial
            category_map = {
                1: "factual",
                2: "reasoning", 
                3: "conversational",
                4: "counterfactual",
                5: "unanswerable",
            }
            category = category_map.get(category_num, f"category_{category_num}")
            questions.append(
                Question(
                    id=f"{conv.get('sample_id', f'conv{idx}')}-q{qi}",
                    text=qa["question"],
                    answer=None if ans is None else str(ans),
                    gold_event_ids=[e for e in evidence if isinstance(e, str)],
                    category=category,
                )
            )
        samples.append(
            Sample(
                id=str(conv.get("sample_id", f"conv{idx}")),
                events=events,
                questions=questions,
            )
        )
    return Dataset(name="locomo10", samples=samples)


def multiday() -> Dataset:
    """A synthetic multi-session stream with EVOLVING facts — the long-horizon failure detector.

    Some facts are *updated* in later sessions (the old value lingers in history); some questions
    are *unanswerable* (the agent should abstain, not confabulate). Filler chatter creates
    retrieval pressure so not everything fits a tight budget. Question categories:
      - 'current'      : answer = the latest value; `stale_answer` = the outdated value
      - 'stable'       : a fact that never changed
      - 'unanswerable' : never stated — the agent should say it doesn't know
    """
    F = "chat"  # filler kind shorthand
    events = [
        # ---- day 1: OLD launch date stated PROMINENTLY (imp5, keyword "launch"); OLD lead ----
        Event("d1-1", "The launch is scheduled for September 14.", kind="fact", importance=5, metadata={"session": "day1"}),
        Event("d1-2", "The primary database is PostgreSQL.", kind="constraint", importance=5, metadata={"session": "day1"}),
        Event("d1-3", "Mark your calendars: September 14 is launch day.", kind="fact", importance=4, metadata={"session": "day1"}),
        Event("d1-4", "The team lead is Mara.", kind="fact", importance=4, metadata={"session": "day1"}),
        Event("d1-5", "We chatted about the weekend football results.", kind=F, metadata={"session": "day1"}),
        Event("d1-6", "Someone mentioned a nice ramen place nearby.", kind=F, metadata={"session": "day1"}),
        # ---- day 2: OLD launch date reinforced a THIRD time ----
        Event("d2-1", "The monthly infrastructure budget ceiling is 2000 euros.", kind="fact", importance=4, metadata={"session": "day2"}),
        Event("d2-2", "Reminder: we're still targeting September 14 for the launch.", kind="fact", importance=4, metadata={"session": "day2"}),
        Event("d2-3", "We debated tabs versus spaces again.", kind=F, metadata={"session": "day2"}),
        Event("d2-4", "Small talk about a vacation to the coast.", kind=F, metadata={"session": "day2"}),
        # ---- day 3: OLD lead reinforced ----
        Event("d3-1", "Mara signed off on the API design.", kind=F, metadata={"session": "day3"}),
        Event("d3-2", "We discussed favourite sci-fi movies.", kind=F, metadata={"session": "day3"}),
        Event("d3-3", "Reminder that the coffee machine is broken.", kind=F, metadata={"session": "day3"}),
        # ---- day 4-5: filler distance between the old facts and their updates ----
        Event("d4-1", "Chatter about the weather turning cold.", kind=F, metadata={"session": "day4"}),
        Event("d4-2", "Someone recommended a podcast about history.", kind=F, metadata={"session": "day4"}),
        Event("d4-3", "Brief talk about lunch options for Friday.", kind=F, metadata={"session": "day4"}),
        Event("d5-1", "Note that parking will be closed next week.", kind=F, metadata={"session": "day5"}),
        Event("d5-2", "Quick aside about the office plants needing water.", kind=F, metadata={"session": "day5"}),
        Event("d5-3", "We argued about the best pizza topping.", kind=F, metadata={"session": "day5"}),
        # ---- day 6: NEW launch date — stated ONCE, paraphrased ("go live"), lower importance, buried ----
        Event("d6-1", "Heads up: we'll actually go live the first week of October instead.", kind="fact", importance=3, metadata={"session": "day6"}),
        Event("d6-2", "Someone shared a meme about Mondays.", kind=F, metadata={"session": "day6"}),
        # ---- day 7: NEW lead — stated once; plus a stable fact ----
        Event("d7-1", "Diego has taken over as team lead from Mara.", kind="fact", importance=3, metadata={"session": "day7"}),
        Event("d7-2", "Decision: API responses must stay under 200ms at p95.", kind="constraint", importance=4, metadata={"session": "day7"}),
        Event("d7-3", "Talk about Friday's team lunch plans.", kind=F, metadata={"session": "day7"}),
    ]
    questions = [
        Question("q-launch", "When is the launch now?", "first week of October", ["d6-1"], category="current", stale_answer="September 14"),
        Question("q-lead", "Who is the current team lead?", "Diego", ["d7-1"], category="current", stale_answer="Mara"),
        Question("q-db", "Which database do we use?", "PostgreSQL", ["d1-2"], category="stable"),
        Question("q-budget", "What is the monthly infrastructure budget?", "2000 euros", ["d2-1"], category="stable"),
        Question("q-latency", "What is the p95 API latency requirement?", "under 200ms", ["d7-2"], category="stable"),
        Question("q-ceo", "What is the name of the CEO?", None, [], category="unanswerable"),
        Question("q-cloud", "Which cloud provider do we use?", None, [], category="unanswerable"),
        Question("q-office", "In which city is the main office?", None, [], category="unanswerable"),
    ]
    return Dataset(name="multiday-v1-adversarial", samples=[Sample("md1", events, questions)])


def conflicts() -> Dataset:
    """Adversarial near-duplicates + temporal conflicts — where local embedding memory usually cheats.

    Three trap patterns, each a known failure mode for embedding-similarity memory:
      - SAME FACT, DIFFERENT DATE: the old value is repeated with near-identical phrasing
        (prominent, high importance); the new value appears ONCE, later, lower importance.
      - SAME PLAN, ONE CONSTRAINT CHANGED: a multi-clause plan is restated verbatim except for
        one number — cosine similarity is ~1, so naive dedup/supersession can pick the wrong copy.
      - ENTITY-CONFUSABLE NEAR-DUPLICATES: two facts that differ only in the entity (Apollo vs
        Artemis). Both are simultaneously TRUE — wrongly superseding one with the other, or
        retrieving the wrong-entity line, is the failure.

    Question categories reuse the multiday machinery: 'current' questions carry the outdated value
    in `stale_answer`; 'confusable' questions carry the WRONG-ENTITY value in `stale_answer` (so
    ctx_stale measures the trap surfacing alone in context); 'stable' are controls; 'unanswerable'
    should produce abstention. All deterministic — no LLM needed via --context-only.
    """
    F = "chat"  # filler kind shorthand
    events = [
        # ---- day 1: temporal-conflict facts stated PROMINENTLY; confusable pair introduced ----
        Event("c1-1", "The quarterly security review is scheduled for March 3.", kind="fact", importance=5, metadata={"session": "day1"}),
        Event("c1-2", "Project Apollo uses PostgreSQL as its database.", kind="fact", importance=4, metadata={"session": "day1"}),
        Event("c1-3", "Project Artemis uses MySQL as its database.", kind="fact", importance=4, metadata={"session": "day1"}),
        Event("c1-4", "Deploy plan: blue-green rollout, 5 percent canary, abort if the error rate exceeds 2 percent.", kind="constraint", importance=5, metadata={"session": "day1"}),
        Event("c1-5", "We chatted about the new espresso machine in the kitchen.", kind=F, metadata={"session": "day1"}),
        # ---- day 2: old values REPEATED near-verbatim (the adversarial reinforcement) ----
        Event("c2-1", "Reminder: the quarterly security review is scheduled for March 3.", kind="fact", importance=4, metadata={"session": "day2"}),
        Event("c2-2", "Deploy plan recap: blue-green rollout, 5 percent canary, abort if the error rate exceeds 2 percent.", kind="constraint", importance=4, metadata={"session": "day2"}),
        Event("c2-3", "Log retention policy: keep application logs for 90 days.", kind="constraint", importance=4, metadata={"session": "day2"}),
        Event("c2-4", "Someone shared photos from the weekend hike.", kind=F, metadata={"session": "day2"}),
        # ---- day 3: second confusable pair (pets) + more reinforcement of old date ----
        Event("c3-1", "Luna, the office cat, has her vet appointment on Friday.", kind="fact", importance=3, metadata={"session": "day3"}),
        Event("c3-2", "Nova, the office dog, has her vet appointment on Monday.", kind="fact", importance=3, metadata={"session": "day3"}),
        Event("c3-3", "Don't forget the March 3 security review when planning the sprint.", kind="fact", importance=3, metadata={"session": "day3"}),
        Event("c3-4", "We debated the best keyboard switches for a while.", kind=F, metadata={"session": "day3"}),
        # ---- day 4: filler distance before the updates ----
        Event("c4-1", "Chatter about the weather and weekend plans.", kind=F, metadata={"session": "day4"}),
        Event("c4-2", "Someone recommended a documentary about volcanoes.", kind=F, metadata={"session": "day4"}),
        Event("c4-3", "Quick aside about the parking garage being full again.", kind=F, metadata={"session": "day4"}),
        # ---- day 5: the UPDATES — each stated ONCE, lower importance, near-identical phrasing ----
        Event("c5-1", "Update: the quarterly security review has moved to March 17.", kind="fact", importance=3, metadata={"session": "day5"}),
        Event("c5-2", "Deploy plan change: blue-green rollout, 5 percent canary, abort if the error rate exceeds 1 percent.", kind="constraint", importance=3, metadata={"session": "day5"}),
        Event("c5-3", "New log retention policy: keep application logs for 30 days.", kind="constraint", importance=3, metadata={"session": "day5"}),
        Event("c5-4", "Someone brought churros for the whole team.", kind=F, metadata={"session": "day5"}),
        # ---- day 6: stable controls + filler ----
        Event("c6-1", "The on-call rotation handbook lives in the team wiki.", kind="fact", importance=3, metadata={"session": "day6"}),
        Event("c6-2", "Decision: all public APIs must be versioned under /v2.", kind="constraint", importance=4, metadata={"session": "day6"}),
        Event("c6-3", "Talk about a new lunch spot opening across the street.", kind=F, metadata={"session": "day6"}),
    ]
    questions = [
        # Temporal conflicts (old value repeated 3x / 2x, new value once)
        Question("q-review", "When is the quarterly security review?", "March 17", ["c5-1"], category="current", stale_answer="March 3"),
        Question("q-abort", "At what error rate do we abort a deploy?", "1 percent", ["c5-2"], category="current", stale_answer="2 percent"),
        Question("q-logs", "How long do we keep application logs?", "30 days", ["c5-3"], category="current", stale_answer="90 days"),
        # Entity-confusable near-duplicates (BOTH sides asked: wrong supersession fails one of them)
        Question("q-artemis", "Which database does Project Artemis use?", "MySQL", ["c1-3"], category="confusable", stale_answer="PostgreSQL"),
        Question("q-apollo", "Which database does Project Apollo use?", "PostgreSQL", ["c1-2"], category="confusable", stale_answer="MySQL"),
        Question("q-nova", "When is Nova's vet appointment?", "Monday", ["c3-2"], category="confusable", stale_answer="Friday"),
        Question("q-luna", "When is Luna's vet appointment?", "Friday", ["c3-1"], category="confusable", stale_answer="Monday"),
        # Stable controls
        Question("q-wiki", "Where does the on-call rotation handbook live?", "team wiki", ["c6-1"], category="stable"),
        Question("q-api", "How must public APIs be versioned?", "/v2", ["c6-2"], category="stable"),
        # Calibration
        Question("q-cdn", "Which CDN provider do we use?", None, [], category="unanswerable"),
        Question("q-region", "In which cloud region is production deployed?", None, [], category="unanswerable"),
    ]
    return Dataset(name="conflicts-v1", samples=[Sample("cf1", events, questions)])


def longmemeval(
    path: str | Path | None = None,
    *,
    max_questions: int | None = None,
    variant: str = "s",
    seed: int = 0,
    abstention_only: bool = False,
) -> Dataset:
    """Load LongMemEval benchmark (ICLR 2025).

    LongMemEval tests long-term memory across 5 abilities: information extraction,
    multi-session reasoning, temporal reasoning, knowledge updates, abstention.

    Each instance -> one Sample: chat history turns become Event[]; the question
    becomes a Question with gold_event_ids = turns marked with has_answer=true.
    Abstention questions (question_id ending with '_abs') have answer=None.

    Args:
        path: Path to the LongMemEval JSON file. If None, tries data/longmemeval_{variant}.json
        max_questions: Limit the number of questions/instances to load.
        variant: Dataset variant - 's' (small, ~115k tokens), 'm' (medium, ~500 sessions),
                 or 'oracle' (evidence sessions only).

    Returns:
        Dataset with samples corresponding to LongMemEval instances.
    """
    default_files = {
        "s": _DATA_DIR / "longmemeval_s.json",
        "m": _DATA_DIR / "longmemeval_m.json",
        "oracle": _DATA_DIR / "longmemeval_oracle.json",
    }
    
    if path is None:
        path = default_files.get(variant)
        if path is None:
            raise ValueError(f"Unknown variant: {variant}. Choose from: s, m, oracle")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"LongMemEval not found at {path}. Download the dataset from "
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned "
            "and place it in the data/ directory."
        )
    
    # Abstention questions (id ends "_abs") are a 30/500 minority scattered through the file, so a
    # first-N stream would miss them — load the full file and filter to measure the *Calibrated* C.
    if abstention_only:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw = [r for r in raw if str(r.get("question_id", "")).endswith("_abs")
               or r.get("question_type") == "abstention"]
    elif max_questions:
        raw = _load_json_sample(path, max_questions=max_questions, seed=seed)
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))

    # Instances are grouped by question_type; shuffle (seeded) so a max_questions cap
    # yields a representative mix across types, not just the first group.
    if max_questions and max_questions < len(raw):
        random.Random(seed).shuffle(raw)
        raw = raw[:max_questions]

    samples: list[Sample] = []
    for idx, instance in enumerate(raw):
        # Build events from haystack_sessions
        events: list[Event] = []
        session_ids = instance.get("haystack_session_ids", [])
        sessions = instance.get("haystack_sessions", [])
        session_dates = instance.get("haystack_dates", [])

        for sess_idx, session in enumerate(sessions):
            session_id = session_ids[sess_idx] if sess_idx < len(session_ids) else f"session_{sess_idx}"
            # All turns in a session share the session's timestamp (event time). Minute-spaced
            # turn offsets preserve in-session order so chronological context stays stable.
            session_ts = _parse_lme_date(session_dates[sess_idx] if sess_idx < len(session_dates) else None)
            for turn_idx, turn in enumerate(session):
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                has_answer = turn.get("has_answer", False)

                event_id = f"{session_id}:{turn_idx}"
                events.append(
                    Event(
                        id=event_id,
                        content=f"{role}: {content}",
                        kind="chat",
                        speaker=role,
                        metadata={
                            "session_id": session_id,
                            "turn_idx": turn_idx,
                            "has_answer": has_answer,
                            "timestamp": (session_ts + turn_idx * 60.0) if session_ts is not None else None,
                        },
                    )
                )

        # Build questions
        questions: list[Question] = []
        question_id = instance.get("question_id", f"q{idx}")
        question_text = instance.get("question", "")
        answer = instance.get("answer")
        question_type = instance.get("question_type", "fact")
        
        # Determine if this is an abstention question
        is_abstention = question_id.endswith("_abs") or question_type == "abstention"
        if is_abstention:
            answer = None
        
        # Collect gold event IDs from turns marked with has_answer=true
        gold_event_ids: list[str] = []
        answer_session_ids = instance.get("answer_session_ids", [])
        
        # Find events that are in answer sessions and have has_answer=true
        for event in events:
            # Check if this event's session is in answer_session_ids
            event_session_id = event.metadata.get("session_id", "")
            if event_session_id in answer_session_ids:
                # Also check if the turn has has_answer=true
                if event.metadata.get("has_answer", False):
                    gold_event_ids.append(event.id)
        
        # If no events with has_answer=true found, try to find all events in answer sessions
        if not gold_event_ids and answer_session_ids:
            for event in events:
                event_session_id = event.metadata.get("session_id", "")
                if event_session_id in answer_session_ids:
                    gold_event_ids.append(event.id)

        # Map LongMemEval question types to our schema categories
        category_map = {
            "single-session-user": "fact",
            "single-session-assistant": "fact",
            "single-session-preference": "preference",
            "multi-session": "multi-session",
            "temporal-reasoning": "temporal",
            "knowledge-update": "current",  # knowledge updates are about current state
            "abstention": "unanswerable",
        }
        category = category_map.get(question_type, "fact")
        
        asked_at = _parse_lme_date(instance.get("question_date"))
        questions.append(
            Question(
                id=question_id,
                text=question_text,
                answer=(str(answer) if answer is not None else None),  # LongMemEval answers can be int
                gold_event_ids=gold_event_ids,
                category=category,
                metadata={"asked_at": asked_at} if asked_at is not None else {},
            )
        )
        
        samples.append(
            Sample(
                id=question_id,
                events=events,
                questions=questions,
            )
        )
    
    variant_name = {"s": "s", "m": "m", "oracle": "oracle"}.get(variant, variant)
    return Dataset(name=f"longmemeval-{variant_name}", samples=samples)


def _load_json_sample(path: Path, *, max_questions: int, seed: int) -> list[dict]:
    """Load a deterministic sample from a large top-level JSON array.

    LongMemEval `s` is hundreds of MB. Reading the whole file as one Unicode string can
    exhaust memory on constrained Windows runs, so for capped evals we stream array items
    through JSONDecoder and keep a small reservoir sample.
    """
    rng = random.Random(seed)
    decoder = json.JSONDecoder()
    reservoir: list[dict] = []
    seen = 0
    buffer = ""
    in_array = False
    done = False

    with path.open("r", encoding="utf-8") as fh:
        while not done:
            chunk = fh.read(1024 * 1024)
            if not chunk and not buffer.strip():
                break
            buffer += chunk
            pos = 0
            while True:
                pos = _skip_json_ws(buffer, pos)
                if not in_array:
                    if pos >= len(buffer):
                        break
                    if buffer[pos] != "[":
                        raise ValueError(f"Expected top-level JSON array in {path}")
                    in_array = True
                    pos += 1
                    continue

                pos = _skip_json_ws(buffer, pos)
                if pos >= len(buffer):
                    break
                if buffer[pos] == ",":
                    pos += 1
                    continue
                if buffer[pos] == "]":
                    done = True
                    pos += 1
                    break
                try:
                    item, next_pos = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    break
                seen += 1
                if len(reservoir) < max_questions:
                    reservoir.append(item)
                else:
                    replace_at = rng.randrange(seen)
                    if replace_at < max_questions:
                        reservoir[replace_at] = item
                pos = next_pos

            buffer = buffer[pos:]
            if not chunk:
                if not done:
                    raise ValueError(f"Unexpected end of JSON while reading {path}")
                break

    rng.shuffle(reservoir)
    return reservoir


def _skip_json_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    return pos
