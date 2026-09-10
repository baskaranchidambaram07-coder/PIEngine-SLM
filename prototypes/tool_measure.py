"""Measure tool-call rate for retrieval-as-a-tool.

Three classes of input:
  SHOULD_CALL  - real questions about the KB. A miss = ungrounded answer (silent
                 wrong answer, the dangerous failure).
  SHOULD_NOT   - greetings / small talk. A call = the cost we were trying to save.
  OFFTOPIC     - real questions the KB cannot answer. Calling is acceptable;
                 what matters is that the model then declines instead of inventing.
"""
import sys
sys.path.insert(0, r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--slm\b122538e-5607-4da6-88f8-92e0cf1c1423\scratchpad")
from tool_rag import run, LEAK   # noqa: E402

GEN = {"temperature": 0.3, "top_p": 0.8, "max_tokens": 512}
MODE = sys.argv[1] if len(sys.argv) > 1 else "agent"

SUITES = {
    "product-insights": {
        "should_call": [
            ("How much does the foldable iPhone cost?", ["1,999", "1999"]),
            ("Which chip is in the iPhone 18 Pro?", ["a20"]),
            ("What changed in Apple Watch health tracking?", ["heart", "health age", "60"]),
            ("What is the cheapest product announced?", ["129", "airpods"]),
            ("When do AirPods 5 ship?", ["18 september", "september 18"]),
            ("How long does the Pro Max battery last?", ["30 hour", "45 hour"]),
        ],
        "should_not": ["hi", "thanks!", "good morning", "ok"],
        "offtopic": ["what is the weather in Chennai today?", "did Apple announce a MacBook Pro?"],
    },
    "meeting-intelligence": {
        "should_call": [
            ("When is Phase 2 go-live?", ["30 september"]),
            ("Who is the executive sponsor at Acme?", ["priya"]),
            ("What SSO approach did we agree?", ["oidc"]),
            ("How much of the Phase 2 budget is consumed?", ["61"]),
            ("Who runs the InfoSec review?", ["anita"]),
            ("When does the master services agreement run to?", ["march 2027"]),
        ],
        "should_not": ["hello", "thank you", "hey there", "ok"],
        "offtopic": ["what is the weather in Chennai today?", "what is the Phase 4 budget?"],
    },
}

totals = {"call_hit": 0, "call_n": 0, "false_call": 0, "nocall_n": 0,
          "grounded": 0, "leak": 0, "reform": 0}

for agent, suite in SUITES.items():
    print("=" * 80)
    print(agent)
    print("=" * 80)

    print("\n  SHOULD CALL (a miss = ungrounded answer)")
    for q, needles in suite["should_call"]:
        called, searched, ans, secs = run(agent, q, GEN, mode=MODE)
        low = ans.lower()
        correct = any(n in low for n in needles)
        reform = bool(searched) and searched[0].strip().lower() != q.strip().lower()
        totals["call_n"] += 1
        totals["call_hit"] += called
        totals["grounded"] += correct
        totals["reform"] += reform
        totals["leak"] += bool(LEAK.search(ans))
        print(f"    {'CALL' if called else 'MISS'}  {'ok ' if correct else 'BAD'}  "
              f"{secs:5.1f}s  {q[:44]:46}")
        if searched:
            print(f"           searched: {searched!r}")
        print(f"           answer  : {ans[:110]!r}")

    print("\n  SHOULD NOT CALL (a call = the cost we are trying to avoid)")
    for q in suite["should_not"]:
        called, searched, ans, secs = run(agent, q, GEN, mode=MODE)
        totals["nocall_n"] += 1
        totals["false_call"] += called
        print(f"    {'CALL(bad)' if called else 'quiet    '}  {secs:5.1f}s  {q:14} -> {ans[:70]!r}")

    print("\n  OFF-TOPIC (calling is fine; inventing an answer is not)")
    for q in suite["offtopic"]:
        called, searched, ans, secs = run(agent, q, GEN, mode=MODE)
        print(f"    {'CALL' if called else 'quiet'}  {secs:5.1f}s  {q[:44]:46} -> {ans[:80]!r}")
    print()

print("=" * 80)
print("HEADLINE NUMBERS")
print("=" * 80)
c, n = totals["call_hit"], totals["call_n"]
fc, fn = totals["false_call"], totals["nocall_n"]
print(f"  tool-call rate on real questions : {c}/{n}  ({100*c/max(n,1):.0f}%)   <- must be ~100%")
print(f"  answered correctly               : {totals['grounded']}/{n}  ({100*totals['grounded']/max(n,1):.0f}%)")
print(f"  false calls on small talk        : {fc}/{fn}  ({100*fc/max(fn,1):.0f}%)")
print(f"  query reformulated (not echoed)  : {totals['reform']}/{n}")
print(f"  malformed/leaked tool syntax     : {totals['leak']}/{n}")
