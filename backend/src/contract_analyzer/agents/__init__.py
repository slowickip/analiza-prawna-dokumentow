"""The worksheet-based multi-agent unit graph.

One call unit is analysed by three acting roles -- a Researcher, an Analyst and
a Verifier -- that read and write one shared worksheet through tools, plus a
Synthesizer that groups the findings of the whole run. A deterministic scheduler
reads the worksheet and names the next role and task, so the routing is
content-driven and reproducible rather than a fixed pipeline. An Explainer
answers the reader afterwards and is not part of any measurement.

The package is split by concern: :mod:`worksheet` owns the shared per-unit
record, :mod:`scheduler` decides who acts next, :mod:`tools` owns the tool
surface and the parity token derived from it, :mod:`turns` owns one bounded tool
conversation, :mod:`analyst` and :mod:`verifier` own what each role may do,
:mod:`findings` turns a decision into a stored finding, and
:mod:`runtime`/:mod:`runner` own the run, which the API, the scripts and the
tests compose from these modules.
"""
