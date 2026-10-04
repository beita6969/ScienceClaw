# Integrated benchmark protocol

## Candidate loop

`load_train -> candidate tool/skill/operator -> submit -> replay -> score_dev`
should be the default visible workflow. The agent can compare frozen tools and
CPU baselines on visible data, then let the evolution engine attribute a passing
trajectory to a skill or operator bundle. Keep the incumbent and candidate
snapshots separate.

## Promotion evidence

A promotion record should contain: dataset code, split, item ids or episode
ids, tool refs and versions, model/checkpoint hashes, config hash, program
snapshot, visible metric, hard-constraint result, replay result, token/time
budget, and whether any external training-data overlap exists.

## Stop conditions

Stop a route when the required frozen asset is unavailable, its input/output
contract cannot be met, formal capacity is exhausted, or a visible comparison
is below the incumbent after a fair fixed comparison. Record the blocker and
move to another dataset rather than tuning against hidden evaluation items.
