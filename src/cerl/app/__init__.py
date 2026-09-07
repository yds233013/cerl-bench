"""The interactive workspace: a thin HTTP adapter over the simulator.

Two servers, deliberately separate processes rather than two routers in one:

* **operational** (:8000) -- what a support agent uses. It serves only what the
  ordinary environment interface exposes, and imports nothing privileged.
* **reviewer** (:8001) -- recorded-episode replay, including verdicts. Launched
  by its own command.

Hiding a privileged panel in the operational app would not be enough: the data
would still be on the wire. So the operational server has no code path to a
verdict, a branch label or a violation flag at all.
"""
