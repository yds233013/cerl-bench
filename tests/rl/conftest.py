"""Model-dependent tests are opt-in.

They load real weights, run real forward passes and take minutes. Nothing in
ordinary development or CI should pay that, so they are skipped unless asked
for explicitly -- by selecting the marker (``-m model``) or by setting
``CERL_RL_MODEL_TESTS=1``.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("CERL_RL_MODEL_TESTS") == "1":
        return
    if "model" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(
        reason="needs model weights; opt in with -m model or CERL_RL_MODEL_TESTS=1",
    )
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)
