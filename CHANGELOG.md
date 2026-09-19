# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- *(tools)* [**breaking**] `send` takes a required `policy` — `try` (the
  original behaviour), `queued` (park the prompt, fire it as the next turn),
  or `steer` (inject into the running turn) ([#34](https://github.com/lexoliu/acpsub/issues/34))
- *(tools)* `wait` and `status` report the current turn number and the
  queued prompt list ([#34](https://github.com/lexoliu/acpsub/issues/34))

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
