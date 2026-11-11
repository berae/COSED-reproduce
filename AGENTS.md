# Research collaboration

Read `docs/START_HERE.md` first. This is a private independent reconstruction and research archive, not the official COSED implementation.

- Root code is the current COSED implementation. `experiments/` contains historical configurations, source snapshots and results; do not silently overwrite historical evidence.
- Establish the research question before proposing architectural changes. Distinguish observations, hypotheses and controlled causal evidence. Negative results are useful.
- Identify checkpoint, split, metric and annotation policy before comparing numbers. DESED Public Test and Validation are separate; MAESTRO here is development; TACOS exact and semantic are separate results.
- A high/low score is not a calibrated probability. Select operational thresholds on validation data, then freeze them for tests. Historical candidate flags are not verified errors.
- Preserve original predictions on the servers. Reuse existing scoring and analysis entry points; avoid creating competing summaries for each change.
- Append meaningful changes to `research/analysis/maintenance.log`. Keep paper-style tables and figures simple, with clear units and definitions.
- Archived scripts retain server paths and may start jobs or downloads. Reading the archive does not require launching them. Check paths, environments and intended experiment before execution.
- Audio, checkpoints and most frame arrays are deliberately not in Git. See inventories and `docs/ARCHIVE.md`; do not invent missing measurements.
- Third-party code keeps its original attribution and terms. This repository does not grant a new blanket license over that material.
