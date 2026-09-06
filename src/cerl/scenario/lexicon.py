"""Synthetic name pools, one disjoint shard per partition.

All data is invented. Shards are disjoint so that a "rename the entities"
perturbation is genuinely free for a policy that learned the rule rather than
the tokens -- the noise floor for every other counterfactual axis -- and so an
agent cannot recognise an evaluation world by the companies in it.

**Shard version 2.0.0.** Corpus 1.x drew every scenario from a single ``core``
pool regardless of partition, so the pools were disjoint and the corpus was not.
Disjointness of a *definition* is necessary and nowhere near sufficient: the
property that matters is that no name materialised into a training file also
appears in an evaluation file, and that is asserted over the committed files in
``tests/scenarios/test_corpus_lexicon.py``.

Shards are keyed by partition name so a shard cannot be chosen without naming
the partition it belongs to. Every value an agent can read is sharded -- company
names, email domains, and staff handles and display names -- because any of them
would serve equally well as a memorisable partition marker.

The 1.x pools are kept below as ``LEGACY_*`` for provenance. They are not
selectable by the generator.
"""

from __future__ import annotations

from cerl.core import FrozenMap

COMPANY_SHARDS: FrozenMap[str, tuple[str, ...]] = FrozenMap(
    {
        "train": (
            "Northwind Traders",
            "Contoso Manufacturing",
            "Fabrikam Logistics",
            "Tailspin Aviation",
            "Wingtip Optics",
            "Litware Analytics",
            "Proseware Robotics",
            "Adventure Works Outfitters",
            "Coho Vineyards",
            "Lucerne Publishing",
            "Woodgrove Financial",
            "Margie's Travel",
        ),
        "validation": (
            "Alpine Ceramics",
            "Bayview Instruments",
            "Cedarline Freight",
            "Dunmore Textiles",
            "Everglade Systems",
            "Foxglove Media",
            "Granite Peak Tools",
            "Harborlight Foods",
            "Ironwood Supply",
            "Junipero Software",
            "Kestrel Dynamics",
            "Lanternhouse Design",
        ),
        "evaluation": (
            "Maplecross Energy",
            "Nightingale Labs",
            "Orchard Bay Trading",
            "Pinewright Interiors",
            "Quarrystone Materials",
            "Riverbend Packaging",
            "Saltmarsh Marine",
            "Thistledown Apparel",
            "Umberfield Print",
            "Vellum & Co",
            "Westmoor Chemical",
            "Yarrowgate Hydraulics",
        ),
    },
)

#: Second-level domain labels, disjoint per partition. In 1.x these three were
#: *identical* -- a "shard" that shards nothing. Every address stays inside the
#: reserved ``.example`` TLD, which is shared on purpose: it is what marks the
#: data as synthetic and it carries no partition information.
DOMAIN_SHARDS: FrozenMap[str, str] = FrozenMap(
    {
        "train": "tr-corp.example",
        "validation": "va-corp.example",
        "evaluation": "ev-corp.example",
    },
)

#: Staff handles and display names, disjoint per partition. Scenario code refers
#: to a stable role name, so it never has to know what that person is called in
#: this shard.
STAFF_SHARDS: FrozenMap[str, FrozenMap[str, tuple[str, str]]] = FrozenMap(
    {
        "train": FrozenMap(
            {
                "agent": ("tr-agent", "Robin Adler"),
                "manager": ("tr-manager", "Dana Whitfield"),
                "director": ("tr-director", "Owen Castellan"),
                "teammate": ("tr-teammate", "Priya Nandakumar"),
                "ops_lead": ("tr-ops-lead", "Marta Oyelaran"),
                "analyst": ("tr-analyst", "Felix Brandt"),
                "trust_lead": ("tr-trust-lead", "Ingrid Solheim"),
            },
        ),
        "validation": FrozenMap(
            {
                "agent": ("va-agent", "Sasha Bellweather"),
                "manager": ("va-manager", "Tomas Lindqvist"),
                "director": ("va-director", "Amara Nwosu"),
                "teammate": ("va-teammate", "Jonah Petrakis"),
                "ops_lead": ("va-ops-lead", "Coretta Villalba"),
                "analyst": ("va-analyst", "Devon Achterberg"),
                "trust_lead": ("va-trust-lead", "Hana Yamashiro"),
            },
        ),
        "evaluation": FrozenMap(
            {
                "agent": ("ev-agent", "Nadia Corriveau"),
                "manager": ("ev-manager", "Rupert Ashgrove"),
                "director": ("ev-director", "Selma Okonkwo"),
                "teammate": ("ev-teammate", "Caleb Marchetti"),
                "ops_lead": ("ev-ops-lead", "Yusuf Demirkan"),
                "analyst": ("ev-analyst", "Willa Thornquist"),
                "trust_lead": ("ev-trust-lead", "Bo Kristiansen"),
            },
        ),
    },
)

#: The 1.x pool names, recorded for provenance. Not selectable by the generator.
LEGACY_SHARD_NAMES: tuple[str, ...] = ("core", "eval_a", "eval_b")

SHARD_VERSION = "2.0.0"


def shard_names() -> tuple[str, ...]:
    return tuple(sorted(COMPANY_SHARDS))


def companies(shard: str) -> tuple[str, ...]:
    """Company names for one partition's shard.

    No default value. A default shard is exactly how corpus 1.x came to draw
    every scenario from ``core``: the parameter existed, and no caller had to
    pass it.
    """
    if shard not in COMPANY_SHARDS:
        raise KeyError(
            f"unknown lexicon shard {shard!r}; expected one of {shard_names()}",
        )
    return COMPANY_SHARDS[shard]


def domain(shard: str) -> str:
    if shard not in DOMAIN_SHARDS:
        raise KeyError(f"unknown lexicon shard {shard!r}")
    return DOMAIN_SHARDS[shard]


def staff(shard: str, role: str) -> tuple[str, str]:
    """``(handle, display_name)`` for a role within one shard."""
    if shard not in STAFF_SHARDS:
        raise KeyError(f"unknown lexicon shard {shard!r}")
    roles = STAFF_SHARDS[shard]
    if role not in roles:
        raise KeyError(f"unknown staff role {role!r} in shard {shard!r}")
    return roles[role]


def all_values(shard: str) -> frozenset[str]:
    """Every lexicon-backed literal this shard can contribute to a file.

    The corpus disjointness test uses this to check that the pool definitions
    and the generated contents actually agree.
    """
    values = set(companies(shard))
    values.add(domain(shard))
    for handle, display in STAFF_SHARDS[shard].values():
        values.add(handle)
        values.add(display)
    return frozenset(values)


def slugify(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in name]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")
