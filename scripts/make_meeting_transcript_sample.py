"""Generate a synthetic meeting-transcript PDF for testing the Meeting Minutes
agent's whole-document analysis mode.

Entirely fictional (Acme Corp is this project's existing demo customer, same
as sample_docs/*.md) and long enough to clear FULLTEXT_CHARS (3500) so a real
run exercises the map-reduce digest path, not the trivial whole-inject one.
Contains concrete decisions, named people, dates, and one explicit follow-up
commitment, so a correct run can be checked against real expectations rather
than "something happened."

    venv\\Scripts\\python scripts\\make_meeting_transcript_sample.py

Writes sample_docs/attachments/meeting-transcript-acme-phase2.pdf.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_docs" / "attachments" / "meeting-transcript-acme-phase2.pdf"

TRANSCRIPT = [
    "ACME CORP — PHASE 2 STEERING COMMITTEE — FOLLOW-UP REVIEW",
    "15 September 2026, 09:30–10:45. Video call.",
    "Attendees: You (Delivery Manager), Priya Sharma (Acme CIO), Daniel Weber "
    "(Acme PMO), Meera Iyer (Solutions Architect), Anil Kapoor (DR/Infra Lead).",
    "",
    "OPENING AND STATUS UPDATE",
    "You opened the call by summarising progress since the last steering "
    "committee on 1 September. Development for the Phase 2 reporting module is "
    "code-complete. Integration testing against the Acme data warehouse is "
    "eighty percent through its planned cases, with eleven of fourteen test "
    "suites green. Priya asked whether the remaining three suites were on the "
    "critical path; Meera confirmed they cover edge cases in currency "
    "conversion and are not blocking, but should close out before UAT begins.",
    "",
    "GO-LIVE DATE",
    "Daniel raised concern that the original go-live date of 1 October left "
    "no buffer for user acceptance testing given the outstanding test suites. "
    "After discussion, the group decided to move the go-live date from "
    "1 October to 15 October 2026, to give UAT a full two weeks rather than "
    "four days. Priya approved the change and asked that the revised plan be "
    "circulated to the wider steering group by end of week.",
    "",
    "DR FAILOVER TEST RESULTS",
    "Anil presented the results of the disaster-recovery failover test run on "
    "10 September. The failover completed with a twelve-minute recovery time "
    "objective, comfortably within the fifteen-minute SLA agreed in the "
    "contract. One issue surfaced: the read replica in the secondary region "
    "took an extra ninety seconds to accept writes after failover, which Anil "
    "will investigate with the database team. He does not expect this to "
    "affect the RTO commitment but wants it resolved before the production "
    "cutover in October.",
    "",
    "PROCUREMENT AND CHANGE REQUESTS",
    "Priya asked Daniel directly to chase procurement sign-off for change "
    "request CR-118, which has been sitting with Acme's procurement team for "
    "eleven days. Daniel committed to escalating internally and getting "
    "sign-off by this Friday, 18 September. Separately, the vendor support "
    "rota for the new monitoring tool still needs to be confirmed; You agreed "
    "to chase the vendor this week since the current rota expires at the end "
    "of the month.",
    "",
    "RISK REGISTER REVIEW",
    "Meera walked through the top three open risks. The highest-rated risk "
    "remains the dependency on Acme's legacy billing system's undocumented "
    "batch job, which the team has now reverse-engineered but not yet fully "
    "tested against. The second risk, key-person dependency on the one "
    "engineer who understands the currency-conversion logic, was downgraded "
    "after a second engineer completed a knowledge-transfer session last "
    "week. The third risk, Acme's slow environment-provisioning process, is "
    "unchanged and continues to add roughly three days to every environment "
    "refresh; Priya offered to raise this with Acme's infrastructure team "
    "directly.",
    "",
    "BUDGET",
    "Daniel confirmed the Phase 2 budget remains on track, with roughly "
    "eighteen percent of the contingency reserve used against a plan of "
    "twenty-five percent at this stage of the programme. No budget action "
    "was needed this cycle.",
    "",
    "ACTION ITEMS DISCUSSED",
    "The group confirmed the following actions verbally, to be tracked "
    "formally afterward: Daniel to secure procurement sign-off for CR-118 by "
    "Friday 18 September. Anil to investigate the read-replica write delay "
    "and report back before the October cutover. You to confirm the vendor "
    "support rota for the monitoring tool this week. Meera to close out the "
    "three remaining currency-conversion test suites before UAT starts on "
    "1 October.",
    "",
    "PRIOR ACTION ITEM STATUS",
    "Before closing, Daniel reviewed the status of action items from the "
    "1 September meeting. The updated integration test plan was circulated on "
    "schedule. The vendor contract amendment for extended support hours was "
    "signed on 8 September. The one item still outstanding is the security "
    "review of the new reporting module's export feature, which InfoSec has "
    "not yet scheduled; Priya said she would chase InfoSec's manager directly "
    "since this review is now a dependency for the 15 October go-live.",
    "",
    "QUESTIONS AND OPEN DISCUSSION",
    "Meera asked whether the revised 15 October date affected the planned "
    "hypercare period after go-live. You confirmed hypercare would still run "
    "for three weeks post-launch, simply shifted two weeks later, and that "
    "the support team's on-call schedule would be adjusted to match. Anil "
    "asked whether the DR test would be repeated after the read-replica issue "
    "is fixed; the group agreed a second, smaller-scope failover test would "
    "run in the first week of October to confirm the fix, rather than "
    "waiting until after go-live. Daniel asked whether the delayed go-live "
    "had any contractual penalty implications; Priya confirmed there is no "
    "penalty clause tied to the 1 October date specifically, only to the "
    "overall Phase 2 completion date in December, which the two-week shift "
    "does not put at risk.",
    "",
    "COMMUNICATION PLAN",
    "The group agreed that the revised go-live date needed to be communicated "
    "to the wider Acme business stakeholders, not just the steering "
    "committee. Daniel will draft a short stakeholder notice explaining the "
    "two-week shift and the reason for it, and will circulate it for review "
    "before sending. Priya asked that the notice make clear this is a "
    "quality-driven decision to protect UAT time, not a sign of the "
    "programme being behind schedule.",
    "",
    "CLOSING AND NEXT STEPS",
    "Priya thanked the team for the update and asked to reconvene once the "
    "revised go-live plan and the outstanding procurement item had moved. "
    "The group agreed to meet again next Thursday, 24 September, at 3pm, to "
    "review the revised numbers, confirm the CR-118 sign-off, and get formal "
    "confirmation of the new 15 October go-live date. Daniel will send the "
    "calendar invite. The call closed at 10:45.",
]


def main() -> int:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=A4)
    w, h = A4
    y = h - 60
    for line in TRANSCRIPT:
        if not line:
            y -= 8
            continue
        is_heading = line.isupper() or (line == line.upper() and len(line) < 90 and any(ch.isalpha() for ch in line))
        c.setFont("Helvetica-Bold" if is_heading else "Helvetica", 11 if is_heading else 10)
        words = line.split(" ")
        wrapped: list[str] = []
        buf = ""
        max_width = w - 100
        for word in words:
            trial = f"{buf} {word}".strip()
            if c.stringWidth(trial, c._fontname, c._fontsize) > max_width:
                wrapped.append(buf)
                buf = word
            else:
                buf = trial
        if buf:
            wrapped.append(buf)
        for wline in wrapped:
            if y < 60:
                c.showPage()
                c.setFont("Helvetica-Bold" if is_heading else "Helvetica", 11 if is_heading else 10)
                y = h - 60
            c.drawString(50, y, wline)
            y -= 15
        y -= 4
    c.showPage()
    c.save()

    text_chars = sum(len(line) for line in TRANSCRIPT)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes on disk, "
          f"~{text_chars} characters of source text)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
