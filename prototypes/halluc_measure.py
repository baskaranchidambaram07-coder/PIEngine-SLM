"""Measure off-topic hallucination in the tool-RAG prototype.

The dangerous case is not "weather in Chennai" — the model knows it has no
weather data. It is an ABSENT ENTITY that sounds like it belongs: a MacBook Pro
in an Apple-event KB, a Phase 4 budget in a Phase-2 project KB. Vector search
always returns its top_k, so the model sees plausible passages and infers the
subject must be covered.

Run:
  python prototypes/halluc_measure.py off        # no lexical verification
  python prototypes/halluc_measure.py on         # verification in the tool result
  python prototypes/halluc_measure.py on 0.05    # verification + min_p sampling
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tool_rag import run   # noqa: E402

VERIFY = (sys.argv[1].lower() == "on") if len(sys.argv) > 1 else True
MIN_P = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
GEN = {"temperature": 0.3, "top_p": 0.8, "max_tokens": 512, "min_p": MIN_P}

# Absent entities that sound like they belong in each KB.
ABSENT = {
    "product-insights": [
        ("did Apple announce a MacBook Pro?", "macbook"),
        ("what is the price of the iPad Pro?", "ipad"),
        ("when does the Vision Pro 2 ship?", "vision"),
        ("what is the M5 chip used for?", "m5"),
        ("how much does the Apple Car cost?", "car"),
        ("what colours does the HomePod come in?", "homepod"),
    ],
    "meeting-intelligence": [
        ("what is the Phase 4 budget?", "phase 4"),
        ("who is the CTO at Acme?", "cto"),
        ("when is the Phase 3 kickoff meeting?", "kickoff"),
        ("what did we decide about the mobile app?", "mobile app"),
        ("what is Acme's Singapore office address?", "singapore"),
        ("who signed the Phase 5 contract?", "phase 5"),
    ],
}

# A correct answer REFUSES. Anything else asserted a fact about something that is
# not in the KB — whether by claiming it exists or by silently answering about a
# different subject ("HomePod colours" -> the iPhone Duo's colours). Both are the
# same failure to the user, so grade on refuse-vs-assert rather than guessing at
# fabrication phrasing.
REFUSED = re.compile(
    r"not\s+(?:\w+\s+){0,2}(?:mentioned|present|specified|provided|included|listed|found"
    r"|discussed|covered|available|referenced|detailed|stated|described)"
    r"|does\s?n[o']t\s+(?:mention|discuss|contain|provide|include|specify|cover|appear"
    r"|list|address|reference|detail|state|describe)"
    r"|do\s?n[o']t\s+have|does\s+not\s+have"
    r"|no\s+(?:information|mention|record|data|reference|details?)\s"
    r"|(?:isn't|is not|not)\s+in\s+the\s+knowledge\s+base"
    r"|cannot\s+(?:find|provide|determine)|could\s+not\s+find|unable\s+to\s+find"
    r"|nothing\s+(?:about|on)\b",
    re.I,
)

print(f"verification={'ON' if VERIFY else 'OFF'}  min_p={MIN_P}  temp={GEN['temperature']}")
print("=" * 84)

good = bad = unclear = 0
for agent, cases in ABSENT.items():
    print(f"\n{agent}")
    print("-" * 84)
    for q, entity in cases:
        _called, searched, ans, secs = run(agent, q, GEN, mode="toolfirst", verify=VERIFY)
        if REFUSED.search(ans):
            verdict, good = "OK      ", good + 1
        else:
            verdict, bad = "ASSERTED", bad + 1
        print(f"  {verdict} {secs:5.1f}s  {q}")
        print(f"           -> {ans[:150]!r}")

n = good + bad + unclear
print("\n" + "=" * 84)
print(f"  correctly refused : {good}/{n}")
print(f"  ASSERTED anyway    : {bad}/{n}")
print(f"  (asserted = claimed it exists, or answered about a different subject)")
