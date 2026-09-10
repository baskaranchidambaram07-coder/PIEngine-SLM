"""End-to-end check of entity-based verification in the tool-RAG prototype.

Two suites that must BOTH hold — an earlier attempt that checked every content
word passed the first and failed the second (11/12 -> 6/12 correct):

  A. absent entities  -> the model must refuse
  B. real questions   -> the model must still answer

Run:  venv\\Scripts\\python prototypes/final_measure.py [on|off]
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tool_rag import run   # noqa: E402

VERIFY = (sys.argv[1].lower() == "on") if len(sys.argv) > 1 else True
GEN = {"temperature": 0.3, "top_p": 0.8, "max_tokens": 512}

ABSENT = {
    "product-insights": ["did Apple announce a MacBook Pro?",
                         "what is the price of the iPad Pro?",
                         "when does the Vision Pro 2 ship?",
                         "what is the M5 chip used for?",
                         "how much does the Apple Car cost?",
                         "what colours does the HomePod come in?"],
    "meeting-intelligence": ["what is the Phase 4 budget?",
                             "who is the CTO at Acme?",
                             "when is the Phase 3 kickoff meeting?",
                             "what did we decide about the mobile app?",
                             "what is Acme's Singapore office address?",
                             "who signed the Phase 5 contract?"],
}

REAL = {
    "product-insights": [
        ("How much does the foldable iPhone cost?", ["1,999", "1999"]),
        ("Which chip is in the iPhone 18 Pro?", ["a20"]),
        ("What changed in Apple Watch health tracking?", ["heart", "health age", "60"]),
        ("What is the cheapest product announced?", ["129", "airpods"]),
        ("When do AirPods 5 ship?", ["18 september", "september 18"]),
        ("How long does the Pro Max battery last?", ["30 hour", "45 hour"]),
    ],
    "meeting-intelligence": [
        ("When is Phase 2 go-live?", ["30 september"]),
        ("Who is the executive sponsor at Acme?", ["priya"]),
        ("What SSO approach did we agree?", ["oidc"]),
        ("How much of the Phase 2 budget is consumed?", ["61"]),
        ("Who runs the InfoSec review?", ["anita"]),
        ("When does the master services agreement run to?", ["march 2027"]),
    ],
}

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

print(f"verification = {'ON (entity-based)' if VERIFY else 'OFF'}   temp={GEN['temperature']}")

print("\n" + "=" * 84)
print("A. ABSENT ENTITIES — the model must refuse")
print("=" * 84)
refused = tot_a = 0
for agent, qs in ABSENT.items():
    print(f"\n{agent}")
    for q in qs:
        _c, _s, ans, secs = run(agent, q, GEN, mode="toolfirst", verify=VERIFY)
        ok = bool(REFUSED.search(ans))
        refused += ok
        tot_a += 1
        print(f"  {'OK      ' if ok else 'ASSERTED'} {secs:5.1f}s  {q[:46]:48}")
        print(f"           -> {ans[:120]!r}")

print("\n" + "=" * 84)
print("B. REAL QUESTIONS — the model must still answer")
print("=" * 84)
correct = tot_b = 0
for agent, cases in REAL.items():
    print(f"\n{agent}")
    for q, needles in cases:
        _c, _s, ans, secs = run(agent, q, GEN, mode="toolfirst", verify=VERIFY)
        ok = any(n in ans.lower() for n in needles)
        correct += ok
        tot_b += 1
        print(f"  {'ok ' if ok else 'BAD'} {secs:5.1f}s  {q[:46]:48}")
        print(f"      -> {ans[:120]!r}")

print("\n" + "=" * 84)
print(f"  A. absent entities refused : {refused}/{tot_a}")
print(f"  B. real questions correct  : {correct}/{tot_b}")
print("  (baselines: verification OFF = 8/12 refused, 11/12 correct;")
print("   all-content-word verification = 12/12 refused, 6/12 correct)")
