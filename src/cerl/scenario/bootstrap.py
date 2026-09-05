"""Register every authored family exactly once, at package import."""

from __future__ import annotations

_DONE = False


def register_all_families() -> None:
    global _DONE  # noqa: PLW0603 - module-level idempotence flag
    if _DONE:
        return
    from cerl.scenario import axes as w2_axes
    from cerl.scenario import generator as w2_generator
    from cerl.scenario import plan as w2_plan
    from cerl.scenario.families import registry
    from cerl.scenario.families import w1_duplicate_profile as w1
    from cerl.scenario.families import w2_duplicate_charge as w2
    from cerl.scenario.families import w3_suspicious_refund as w3

    registry.register(
        w2.TEMPLATE, w2_generator.generate, w2_plan.all_instances, w2_axes.axes_slug,
    )
    registry.register(w1.TEMPLATE, w1.generate, w1.instances, w1.axes_slug)
    registry.register(w3.TEMPLATE, w3.generate, w3.instances, w3.axes_slug)
    _DONE = True
