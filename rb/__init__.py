# Revision-Bench (RB) v2 — docs/rb_design.md. Evaluation-only benchmark
# machinery: registry (domains/tools/scenarios), grammar (layers/events/
# templates), sandbox (deterministic tools), generator (episode grid), audio
# (TTS backends + timeline assembly), simulator (arm-B reactive user), scorer
# (official-compatible / transactional / utility tracks).
#
# FIREWALL: nothing in this package may feed training, tuning, or selection
# of any system component. RB-test is single-shot per system version.
