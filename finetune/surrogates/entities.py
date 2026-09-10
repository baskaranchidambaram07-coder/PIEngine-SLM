"""Entity pools for instantiating surrogate documents.

Surrogate docs are TEMPLATES, not fixtures. Every time the synthesiser needs a
context it renders a template with a fresh draw from these pools, so the model
sees the same document *shape* carrying different facts on every example.

That is what keeps the adapter behavioural. If the training set contained one
fixed cast, three epochs would teach the model that Nordveil's go-live is
30 September — a fact that is false for every real customer and that RAG was
already going to supply. Rotating the cast makes the only learnable signal the
behaviour: cite the document, refuse what is absent, keep it short.
"""
from __future__ import annotations

import random

FIRST = ["Anita", "Rhea", "Daniel", "Priya", "Marcus", "Leena", "Tobias", "Sana",
         "Ivan", "Meera", "Colin", "Yuki", "Ravi", "Astrid", "Omar", "Nadia",
         "Felix", "Grace", "Hector", "Ingrid", "Jonas", "Kavya", "Lucia", "Noor"]
LAST = ["Okonkwo", "Vasquez", "Lindqvist", "Raman", "Duarte", "Novak", "Bergman",
        "Hassan", "Petrov", "Iyer", "Whitfield", "Tanaka", "Menon", "Kaur",
        "Fontaine", "Oyelaran", "Brandt", "Silva", "Moreau", "Nkemdirim"]
COMPANY = ["Nordveil Logistics", "Calderon Retail Group", "Meridian Health",
           "Quillon Bank", "Arvent Manufacturing", "Sablefield Energy",
           "Tessaro Media", "Highmoor Insurance", "Verdanix Foods"]
PROGRAMME = ["Atlas", "Beacon", "Cobalt", "Driftwood", "Ember", "Foundry",
             "Granite", "Halyard", "Ironwood", "Juniper", "Keystone", "Lantern"]
CITY = ["Rotterdam", "Bengaluru", "Toronto", "Lisbon", "Osaka", "Nairobi",
        "Warsaw", "Melbourne", "Santiago", "Dublin"]
SYSTEM = ["the order management platform", "the claims engine", "the CRM",
          "the settlement gateway", "the warehouse scheduler", "the billing hub"]
VENDOR = ["Solvexa", "Trellick", "Bramhall Systems", "Cintric", "Waypost"]
MONTH = ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"]
RISK = ["a single-threaded integration owner", "unresolved data residency",
        "a hard cutover with no rollback window", "an unratified interface contract",
        "peak-season change freeze overlap", "an untested failover path"]
RAG_STATUS = ["GREEN", "AMBER", "RED"]
TOPIC = ["single sign-on", "data migration", "UAT entry criteria", "the cutover plan",
         "performance testing", "the support model", "invoice reconciliation",
         "role-based access"]


def draw(rng: random.Random) -> dict[str, str]:
    """One consistent cast for one rendered document set."""
    people = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], 6)
    prog = rng.choice(PROGRAMME)
    year = rng.choice([2026, 2027])
    m1, m2 = rng.sample(range(len(MONTH)), 2)
    return {
        "company": rng.choice(COMPANY),
        "programme": f"Programme {prog}",
        "prog": prog,
        "sponsor": people[0], "pm": people[1], "arch": people[2],
        "sec": people[3], "ops": people[4], "vendor_lead": people[5],
        "vendor": rng.choice(VENDOR),
        "city": rng.choice(CITY),
        "system": rng.choice(SYSTEM),
        "topic": rng.choice(TOPIC),
        "risk": rng.choice(RISK),
        "d1": f"{rng.randint(1, 28)} {MONTH[m1]} {year}",
        "d2": f"{rng.randint(1, 28)} {MONTH[m2]} {year}",
        "d3": f"{rng.randint(1, 28)} {MONTH[(m2 + 2) % 12]} {year}",
        "pct1": str(rng.randint(31, 89)),
        "pct2": str(rng.randint(31, 89)),
        "amt1": f"{rng.randint(2, 40)}0,000",
        "n1": str(rng.randint(3, 40)),
        "n2": str(rng.randint(3, 40)),
        "ref1": f"{prog[:2].upper()}-{rng.randint(100, 999)}",
        "ref2": f"{prog[:2].upper()}-{rng.randint(100, 999)}",
        "rag": rng.choice(RAG_STATUS),
    }


def render(text: str, cast: dict[str, str]) -> str:
    for k, v in cast.items():
        text = text.replace("{{" + k + "}}", v)
    return text
