# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/lexoliu/acpsub/compare/v0.2.0...v0.3.0) - 2026-10-03

### Added

- *(daemon)* graceful restart — drain in-flight turns and resume every live session
- park rate-limited turns and resume them when the limit lifts
- *(tools)* --max-wait deadline with an activity digest on wait
- add daemon + CLI client alongside the stdio MCP server
- report MCP progress while wait/wait_any block ([#46](https://github.com/lexoliu/acpsub/pull/46))
- [**breaking**] wait/wait_any take a required expect_secs and report overrun ([#44](https://github.com/lexoliu/acpsub/pull/44))
- [**breaking**] make model and mode required spawn/adopt arguments
- *(tools)* [**breaking**] require an explicit send policy (try|queued|steer) ([#35](https://github.com/lexoliu/acpsub/pull/35))

### Fixed

- *(restart)* surface failed resumes and sequence tests on real signals
- *(restart)* race-free drain signaling, EOF restart waits, structured refusal
- *(cli)* make --model and --mode optional on spawn and adopt
- validate spawn/adopt mode, model, and config against agent advertisements ([#43](https://github.com/lexoliu/acpsub/pull/43))

### Other

- terminate files with a newline
- *(tools)* type the activity digest and require tool-call start times
- Merge remote-tracking branch 'origin/dev' into fix/cli-optional-model-mode
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
