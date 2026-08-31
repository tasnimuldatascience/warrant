Working files for the design canvas, kept so a later change re-seeds rather than restarts.

`Main.dc.html`, `Timeline.dc.html` and `Trace.dc.html` are the three screens — Ask, Timeline,
Trace — and `canvas.json` lays them out. `warrant-interface.html` is the seeded canvas: it is
generated, gitignored, and must not be hand-edited; change a source and re-seed.

These are **mockups**. The interface that actually runs is `ui/`, and the two are allowed to
disagree while a direction is being settled. When they do, `ui/` is the truth — a design file
that quietly stops matching the thing it describes is worse than no design file, because it
is the one a reader trusts.
