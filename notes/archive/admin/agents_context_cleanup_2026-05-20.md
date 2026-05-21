# AGENTS.md Context Cleanup (2026-05-20)

`AGENTS.md` should contain stable agent behavior and durable repo guidance only.
Volatile project state was removed from `AGENTS.md` and should live in notes.

## Moved Out Of AGENTS.md

- Active experiment, current branch, current focus files, pending validation status:
  `notes/current_status.md`
- Slurm commands, wrapper defaults, build commands, environment variables:
  `notes/runbook.md`
- Specific measurements, Slurm job IDs, latency/quality findings, failed ideas:
  `notes/findings_log.md` or a dated experiment note
- Open research/design questions:
  `notes/research_flow.md`
- Historical handoff details:
  dated `notes/context_checkpoint_*.md` files

## Deleted From AGENTS.md

- Stale quick-start commands such as `sbatch test.sh`.
- The old "do not tail/wait after submit" rule, which conflicts with active Slurm monitoring workflows.
- Specific RetrievalAttention graph-builder defaults and Roar decode flags.
- Specific seed-mode A/B results and old Slurm output references.

## Current Principle

If a detail can become stale after one experiment run, it should not be in
`AGENTS.md`. Put it in a dated note, `notes/current_status.md`, or
`notes/runbook.md` instead.
