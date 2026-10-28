# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/lexoliu/acpsub/compare/v0.2.0...v0.3.0) - 2026-10-28

### Added

- say when an unwatched turn will be stopped
- *(daemon)* prune old transcripts and registry entries
- [**breaking**] answer in prose by default; wait returns only the turn's reply ([#108](https://github.com/lexoliu/acpsub/pull/108))
- [**breaking**] replace --owner pid reaping with a wait-fed watchdog
- *(tools)* [**breaking**] report a steer as submitted and resolve it in status and wait ([#71](https://github.com/lexoliu/acpsub/pull/71))
- *(daemon)* graceful restart — drain in-flight turns and resume every live session
- park rate-limited turns and resume them when the limit lifts
- *(tools)* --max-wait deadline with an activity digest on wait
- add daemon + CLI client alongside the stdio MCP server
- report MCP progress while wait/wait_any block ([#46](https://github.com/lexoliu/acpsub/pull/46))
- [**breaking**] wait/wait_any take a required expect_secs and report overrun ([#44](https://github.com/lexoliu/acpsub/pull/44))
- [**breaking**] make model and mode required spawn/adopt arguments
- *(tools)* [**breaking**] require an explicit send policy (try|queued|steer) ([#35](https://github.com/lexoliu/acpsub/pull/35))

### Fixed

- re-issue a dropped idempotent call only against a new daemon
- always serialize a resume report's pending turns
- *(ci)* install the glibc release-plz build for releases ([#111](https://github.com/lexoliu/acpsub/pull/111))
- *(wait)* refuse an --expect the turn has already passed ([#106](https://github.com/lexoliu/acpsub/pull/106))
- *(wait)* list a turn's tool calls in start order ([#103](https://github.com/lexoliu/acpsub/pull/103))
- surface provider rate limits from spawn, send and wait
- *(daemon)* refuse calls for a shipped session with the handoff
- *(daemon)* record a turn's end before reporting it
- *(daemon)* recover when a successor dies after takeover
- hold sends refused mid-handoff until the ship resolves
- *(transcript)* hide session-state updates and count --tail by section
- report a failed socket bind as itself
- resume a session parked before its turn with its own prompt
- refuse a live socket whose daemon holds no claim
- [**breaking**] restart hands the socket over instead of draining
- one daemon per socket, and tests stop the daemons they start ([#75](https://github.com/lexoliu/acpsub/pull/75))
- a close or forget during a launch ends the session when the launch lands
- a wait's progress is its own elapsed time
- a closed session never resumes on its own schedule
- *(restart)* surface failed resumes and sequence tests on real signals
- *(restart)* race-free drain signaling, EOF restart waits, structured refusal
- *(cli)* make --model and --mode optional on spawn and adopt
- resume failed turns and let adopt reap exited agents
- *(wait)* bind wait and wait_any to the turn running at call start
- allow spawn/adopt on agents that advertise no modes or model option
- settle steers a folding agent never answers
- validate spawn/adopt mode, model, and config against agent advertisements ([#43](https://github.com/lexoliu/acpsub/pull/43))

### Other

- Merge pull request #114 from lexoliu/fix/110-restart-hang
- drive rate-limit and park tests by events, not wall-clock windows
- Merge origin/dev into fix/87-restart-wait
- rustfmt
- collect clients by exit, not pipe EOF
- rustfmt
- Merge remote-tracking branch 'origin/dev' into fix/restart-handover
- Merge pull request #78 from lexoliu/fix/71-steer-submitted-r2
- Merge pull request #72 from lexoliu/fix/68-close-in-flight-launch
- Reap whole process groups when sessions and terminals end ([#64](https://github.com/lexoliu/acpsub/pull/64))
- reject expectations over 30 minutes unless --long
- terminate files with a newline
- *(tools)* type the activity digest and require tool-call start times
- Merge remote-tracking branch 'origin/dev' into fix/cli-optional-model-mode
- *(tools)* drop intra-doc link to private PROGRESS_INTERVAL
- wait on the real signal in progress tests
- *(wait)* cover turn binding across a chained queued turn
- Merge remote-tracking branch 'origin/fix/agents-without-modes' into HEAD
- *(tools)* wait on the turn-end signal in the steer-turn test

### Added

- *(cli)* `acpsub daemon` serves the tool set over a Unix socket
  (`~/.local/share/acpsub/daemon.sock`), holding sessions independently of
  any MCP host; every tool is also a client subcommand
  (`acpsub spawn|adopt|send|wait|wait-any|status|result|cancel|permit|list|close|forget|agents-live`),
  and a client auto-starts the daemon when the socket is silent. `wait`
  blocking in a background process replaces a blocking MCP `tools/call`
- *(daemon)* orphan reaping: a subagent spawned with `--owner PID` is closed
  once that pid is dead, so sessions cannot outlive their coordinator
- *(tools)* `spawn`/`adopt` accept an optional `owner` pid
- *(tools)* [**breaking**] `send` takes a required `policy` — `try` (the
  original behaviour), `queued` (park the prompt, fire it as the next turn),
  or `steer` (inject into the running turn) ([#34](https://github.com/lexoliu/acpsub/issues/34))
- *(tools)* `wait` and `status` report the current turn number and the
  queued prompt list ([#34](https://github.com/lexoliu/acpsub/issues/34))
- *(tools)* `wait`/`wait_any` take `max_wait_secs` (`--max-wait`), the
  wait call's own deadline: when it passes before the awaited turn ends
  or overruns, a `running` result carries an activity `digest` of the
  window (call count, wall-time share, five longest calls, repeated
  titles, time since the last edit-kind call, latest call); `overrun`
  results carry the same digest. Every value is measured from tool calls'
  recorded `started_at`/`ended_at`, now reported in the `tool_calls`
  summaries ([#55](https://github.com/lexoliu/acpsub/issues/55))
- *(tools)* turns that fail with a provider's structured rate-limit
  error park in a new `rate_limited` state carrying `resume_at` — the
  reset the message's `(at HH:MM UTC)` clause names, never a guess —
  and resume on their own at that time in the same ACP session
  (`session/load` when the process is gone), staggered on a fixed
  continuation prompt. The quota binds the agent config key, so every
  `send`/`spawn`/`adopt` prompt in the scope parks (`{state:
  "rate_limited", resume_at, reason, position}`) and runs after the
  continuation; `wait` freezes through the pause without counting it
  against `expect_secs`, and `status` reports `resume_at`/`reason`. The
  schedule — and the session's owner — persists in the registry, so
  both survive a daemon restart
  ([#56](https://github.com/lexoliu/acpsub/issues/56))
- *(tools)* `wait`/`wait_any` reject `expect_secs`/`max_wait_secs`
  (`--expect`/`--max-wait`) over 1800 s (30 minutes) unless the call
  passes `long` (`--long`) — a local subagent is meant for short
  execution, so a longer expectation is a planning error the caller must
  own ([#61](https://github.com/lexoliu/acpsub/issues/61))
- *(tools)* [**breaking**] `send` with `policy: "steer"` returns
  `{state: "submitted", steer_id, turn}` at once instead of waiting out
  a 250 ms acknowledgement window for `steered` — ACP has no positive
  acknowledgement for a folded steer, so `send` reports only the
  submission. `status` now lists the session's unresolved steers (id and
  target turn) in `steers`, and a `wait`/`wait_any` bound to a turn holds
  until every steer that targeted it has resolved, reporting each in
  `steers` with its outcome — `folded`, `rejected` with the agent's
  error, or `turn` when it ran as a turn of its own
  ([#71](https://github.com/lexoliu/acpsub/issues/71))

## [0.2.0](https://github.com/lexoliu/acpsub/compare/v0.1.0...v0.2.0) - 2026-09-16

### Added

- *(tools)* [**breaking**] address sessions by agent-assigned session_id; add adopt ([#21](https://github.com/lexoliu/acpsub/pull/21))

### Other

- *(release)* pass --git-token explicitly to release-plz ([#28](https://github.com/lexoliu/acpsub/pull/28))
- *(release)* pass GITHUB_TOKEN to release-plz/git-config ([#27](https://github.com/lexoliu/acpsub/pull/27))
- *(release)* run release-plz release-pr on dev, only release on main ([#26](https://github.com/lexoliu/acpsub/pull/26))
- *(release)* ship prebuilt binaries via cargo-dist ([#23](https://github.com/lexoliu/acpsub/pull/23))
- Reject wait/wait_any timeouts below 60s ([#18](https://github.com/lexoliu/acpsub/pull/18))
- *(tools)* steer wait/wait_any callers to 5-30 min timeouts ([#16](https://github.com/lexoliu/acpsub/pull/16))
- *(tools)* cover agent death, kill, close, send/fs/permit edge cases ([#7](https://github.com/lexoliu/acpsub/pull/7))
- release v0.1.0 ([#12](https://github.com/lexoliu/acpsub/pull/12))

## [0.1.0](https://github.com/lexoliu/acpsub/compare/v0.0.0...v0.1.0) - 2026-09-12

### Other

- *(release)* publish to crates.io via trusted publishing ([#11](https://github.com/lexoliu/acpsub/pull/11))
- Initial commit