"""Synthetic name pools, partitioned into disjoint shards per split.

All data is invented. Shards are disjoint so that a "rename the entities"
perturbation is genuinely free for a policy that learned the rule rather than the
tokens -- the noise floor for every other counterfactual axis.

Phase 1A uses the ``core`` shard only; the others exist so the split machinery in
Phase 1B has disjoint pools to draw from and the disjointness test has something
to assert.
"""

from __future__ import annotations

from cerl.core import FrozenMap

COMPANY_SHARDS: FrozenMap[str, tuple[str, ...]] = FrozenMap(
    {
        "core": (
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
        "eval_a": (
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
        "eval_b": (
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

DOMAIN_SHARDS: FrozenMap[str, tuple[str, ...]] = FrozenMap(
    {
        "core": ("example", "test", "invalid"),
        "eval_a": ("example", "test", "invalid"),
        "eval_b": ("example", "test", "invalid"),
    },
)

STAFF_NAMES: tuple[tuple[str, str], ...] = (
    ("agent", "Support Agent"),
    ("manager", "Billing Manager"),
    ("director", "Finance Director"),
    ("teammate", "Support Teammate"),
)


def companies(shard: str = "core") -> tuple[str, ...]:
    return COMPANY_SHARDS[shard]


def slugify(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in name]
    slug = "".join(keep)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")
