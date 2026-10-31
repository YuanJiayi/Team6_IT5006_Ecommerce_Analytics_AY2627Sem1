# Rules for Claude in this repo (Winnie's Phase 3 work)

- Work only on the `winnie-phase3` branch. Never switch to, commit to, or push to `main` or a teammate's branch.
- Create or edit files only inside `Phase3_Winnie/`. Never modify, rename, move or delete any existing file
  outside it (Phase1/, Phase2/, ref/, README.md, .gitignore, etc.). Reading them is fine.
- The baseline is the Phase 2 final models in `Phase3_Winnie/models/`. Never overwrite them.
- Put each experiment in its own new folder: `Phase3_Winnie/experiments/NN_short_name/`
  (script, results and any new model files). Never overwrite an earlier experiment.
- Build model inputs with `load_cohort` from `Phase2/phase2_handover_classifier.py`; do not compute features by hand.
- Compare every experiment against the Phase 2 baseline on the same test period and metrics as the report.
- Do not commit or push. Show me what changed; I review, commit and push myself.
- If a task seems to need changing a file outside `Phase3_Winnie/`, stop and ask me first.