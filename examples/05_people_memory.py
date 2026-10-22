"""A conversation updates three people you know and introduces a fourth.

``uv run python examples/05_people_memory.py [--offline]``

The scenario is trustcall's "relationships" demo: you keep a list of people, and a
chat mentions several of them at once, plus a stranger and a dog. One call has to
update the right records, insert the new person and ignore the dog.

What to look at:

* ``Records``: ids ``0``, ``1``, ``2`` are updates of the existing records you passed
  in, in order of first mention; Olivia Davis is an insert (``id -``). Each person of
  the conversation is matched to one existing record or to none (one Clef choice per
  person; the assignment is one-to-one), and a record the conversation does not
  mention would be listed as untouched (there are none here).
* ``skipped``: "Sunny" was found as a name but is a dog, so the qualification check
  ("is Sunny a Person?") refuses it: ``not_qualified``.
* Notes are appended with ``add /notes/-``; the existing notes are never rewritten.
* Clairic is extractive: every text value is a verbatim span of the conversation.
  That is why the new notes read like quotes ("got promoted to Senior Marketing
  Manager last month") rather than tidy paraphrases, and why Olivia's relationship
  comes back as "my cousin" in the *friend's* voice. If you need a normalised
  vocabulary, make the field a ``Literal`` (a Clef choice) instead of free text.
* Michael's record is the verification ladder at work: the friend's remark about rock
  climbing sits right before Michael's name, so a reader focused on Michael picks it up;
  the attribution check ("does the input say this about Michael?") drops it
  (``rejected``). Sarah's name comes back as the short form "Sarah", which a naive
  diff would write over "Sarah Johnson": the diff itself refuses to replace an identity
  field with a part of it (``identity_subset``, before any model is asked). The
  relationship "neighbor" is the record's "Neighbor" up to case, which free text does
  not count as a change: no operation, no gate request.
* The match itself is checked twice. Each person is matched to one record by a Clef
  choice (one-to-one over all of them); and a person whose extracted name shares no
  name with the record they were matched to would have the whole update refused
  (``identity_conflict``). "Sarah" and "Sarah Johnson" share one, so it goes through.
* Each mention is shown with its span in context.

Live mode runs on the real models (about forty requests); offline mode scripts the
answers. Either way ``Many`` plus ``existing`` is the whole interface.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    build_extractor,
    cli,
    only_record,
    print_banner,
    resolve_offline,
    show_call_log,
    show_clusters,
    show_mentions,
    show_overview,
    show_patch_lines,
    show_record,
    show_source,
    show_summary,
    show_verification,
)
from _offline import (
    OfflineScript,
    change_gate,
    entity_count,
    mention_discovery,
    nothing,
    pair_coreference,
    qualifier,
    record_matcher,
    says,
)
from pydantic import BaseModel, Field

from trustcall_system_one import ExtractionResult, Many

CONVERSATION = """\
Me: I ran into Emma Thompson at the park yesterday. She was walking her new puppy, a golden retriever named Sunny. She mentioned she got promoted to Senior Marketing Manager last month.
Friend: That's great news for Emma! How's she enjoying the new role?
Me: She seems to be thriving. Oh, and did you know she's taken up rock climbing? She invited me to join her at the climbing gym sometime.
Friend: Wow, rock climbing? That's quite a change from hiking. Speaking of friends, have you heard from Michael Chen recently?
Me: Actually, yes. We had a video call last week. He's switched jobs and is now working as a Data Scientist at a startup. He's also mentioned he's thinking of going vegan.
Friend: That's a big change for Michael! Both career and diet-wise. How about your neighbor, Sarah? Is she still teaching?
Me: Sarah's doing well. Her kids are growing up fast - her oldest just started middle school. She's still teaching, but now she's focusing on special education. She's really passionate about it.
Friend: That's wonderful. Oh, before I forget, I wanted to introduce you to my cousin who just moved to town. Her name is Olivia Davis, she's a 27-year-old graphic designer. She's looking to meet new people and expand her social circle. I thought you two might get along well.
Me: That sounds great! I'd love to meet her. Maybe we could all get together for coffee next week?
Friend: Perfect! I'll set it up. Olivia loves art and is always sketching in her free time. She also volunteers at the local animal shelter on weekends."""  # noqa: E501


class Person(BaseModel):
    """Someone the speaker knows or is being introduced to."""

    name: str = Field(description="full name")
    relationship: str = Field(description="how the person relates to the speaker")
    notes: list[str] = Field(description="things worth remembering about the person")


EXISTING = [
    Person(
        name="Emma Thompson",
        relationship="College friend",
        notes=["Loves hiking", "Works in marketing", "Has a dog named Max"],
    ),
    Person(
        name="Michael Chen",
        relationship="Coworker",
        notes=["Great at problem-solving", "Vegetarian", "Plays guitar"],
    ),
    Person(
        name="Sarah Johnson",
        relationship="Neighbor",
        notes=["Has two kids", "Loves gardening", "Makes amazing cookies"],
    ),
]


def offline_script() -> OfflineScript:
    """What Clairic and Clef would plausibly answer for ``CONVERSATION``."""
    names = [
        "Emma Thompson",
        "Emma",
        "Michael Chen",
        "Michael",
        "Sarah",
        "Olivia Davis",
        "Olivia",
        "Sunny",
    ]
    same_person = {
        "Emma Thompson": "emma",
        "Emma": "emma",
        "Michael Chen": "michael",
        "Michael": "michael",
        "Sarah": "sarah",
        "Olivia Davis": "olivia",
        "Olivia": "olivia",
        "Sunny": "sunny",
    }

    def notes(*found: str | tuple[str, int], confidence: float = 0.9) -> dict[str, Any]:
        return says(
            "list", None, find_all=list(found), p_null=0.03, confidence=confidence
        )

    script = OfflineScript(
        source=CONVERSATION,
        presence=0.99,
        clairic={
            "Emma::name": says("text", "Emma Thompson", find="Emma Thompson"),
            "Emma::relationship": nothing("text", p_null=0.93),
            "Emma::notes": notes(
                "walking her new puppy, a golden retriever named Sunny",
                "got promoted to Senior Marketing Manager last month",
                "taken up rock climbing",
            ),
            "Michael::name": says("text", "Michael Chen", find="Michael Chen"),
            "Michael::relationship": nothing("text", p_null=0.94),
            "Michael::notes": notes(
                "switched jobs and is now working as a Data Scientist at a startup",
                "thinking of going vegan",
                ("rock climbing", 1),  # Emma's: right before Michael's name
            ),
            # Only the short form is in the text.
            "Sarah::name": says("text", "Sarah", find="Sarah", confidence=0.9),
            "Sarah::relationship": says("text", "neighbor", find="neighbor"),
            "Sarah::notes": notes(
                "her oldest just started middle school",
                "focusing on special education",
                "really passionate about it",
            ),
            "Olivia::name": says("text", "Olivia Davis", find="Olivia Davis"),
            "Olivia::relationship": says(
                "text", "my cousin", find="my cousin", confidence=0.88
            ),
            "Olivia::notes": notes(
                "a 27-year-old graphic designer",
                "looking to meet new people and expand her social circle",
                "loves art and is always sketching in her free time",
                "volunteers at the local animal shelter on weekends",
            ),
        },
        bleed=[("Michael", "rock climbing", 0.05)],
    )
    script.handlers["discover"] = mention_discovery(names)
    script.handlers["count"] = entity_count(4)
    script.handlers["coref"] = pair_coreference(same_person)
    script.handlers["qualify"] = qualifier({"Sunny": 0.04})
    script.handlers["match"] = record_matcher(
        {
            "Emma": "Emma Thompson",
            "Michael": "Michael Chen",
            "Sarah": "Sarah Johnson",
            "Olivia": None,
            "Sunny": None,
        }
    )
    script.handlers["gate"] = change_gate(0.94)
    return script


def main(offline: bool = False) -> ExtractionResult:
    """Run the scenario and print what happened to every person; return the result."""
    offline = resolve_offline(offline)
    print_banner(offline)
    script = offline_script() if offline else None
    existing = [(str(i), "Person", p.model_dump()) for i, p in enumerate(EXISTING)]
    extractor = build_extractor(
        [Many(Person, identity="name")],
        offline=offline,
        script=script,
        enable_inserts=True,
    )
    with extractor:
        result = extractor.invoke(CONVERSATION, existing=existing)

    show_source("The conversation", CONVERSATION)
    if not result.records:
        only_record(result, "person")
        show_call_log(result, script)
        return result
    show_overview(result)
    show_clusters(result)
    for record in result.records:
        show_record(record, CONVERSATION, snippets=True)
        if record.kind == "update":
            show_patch_lines(
                record,
                "Each applied operation, next to the text behind it",
                CONVERSATION,
            )
        show_mentions(record, CONVERSATION)
    show_verification(result)
    show_summary(result, offline=offline)
    return result


if __name__ == "__main__":
    cli(main, __doc__)
