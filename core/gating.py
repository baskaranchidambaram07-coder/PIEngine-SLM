"""Decide whether a message should hit the knowledge base at all.

Retrieval used to run unconditionally, so a bare "hi" was embedded, searched,
and answered with four chunks of context (~1000 prompt tokens). The obvious
patch — raise rag.min_score — does not generalise: cosine similarity from a
bi-encoder is a RANKING function, not a relevance detector. Its embeddings are
anisotropic, so unrelated text still scores ~0.45-0.55 and there is no "no
match" value to threshold against. Measured on this project's own two KBs,
"thanks!" (0.5553) outranks the genuine question "what did we decide about the
architecture?" (0.5547), so no threshold separates them.

So the gate here is deterministic and lexical, not score-based. It answers only
the question it can answer safely: does this message contain anything to look
up? A message with no content words, or one made entirely of greeting tokens,
cannot be a knowledge-base query. Anything else is passed through to the
retriever, where min_score still acts as a floor.

This is intentionally conservative — it must never suppress a real question.
Off-topic *questions* ("what is the weather in Chennai?") are NOT caught here;
no lexical rule catches those. Those need the model to decide, i.e. retrieval
exposed as a tool. See docs/rag-gating.md.

android_app/AgentRuntime/src/gating.ts is a line-for-line port of this module.
Change both together or the device and the server will disagree.
"""
from __future__ import annotations

import re

# Function words carry no lookup value on their own.
STOPWORDS = frozenset("""
a an the is are was were be been being am do does did doing have has had having
what which who whom whose when where why how this that these those i you he she
it we they me him her us them my your his her its our their to of in on at for
with about from by as and or but if then than so not no nor too very can could
will would shall should may might must there here all any some more most other
into over under again further once
""".split())

# A message made only of these is social, not a query. Deliberately short:
# every addition is a chance to swallow a real question.
GREETING_WORDS = frozenset("""
hi hello hey heya yo hiya greetings morning afternoon evening night good
thanks thank thankyou thx ty cheers appreciated welcome
ok okay okey k sure yeah yep yes nope nah fine cool nice great awesome perfect
bye goodbye later farewell see ya cya
please sorry apologies oops hmm hm ah oh wow
""".split())

MAX_GREETING_WORDS = 5

_WORD = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with function words removed."""
    return [w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS]


def should_retrieve(query: str) -> tuple[bool, str]:
    """Return (retrieve?, reason). Reason is for the diagnostic log."""
    if not query or not query.strip():
        return False, "empty message"

    words = content_words(query)
    if not words:
        return False, "no content words"

    if len(words) <= MAX_GREETING_WORDS and all(w in GREETING_WORDS for w in words):
        return False, "greeting only"

    return True, "has content"
