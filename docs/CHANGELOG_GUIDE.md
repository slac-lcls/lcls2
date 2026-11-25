# Changelog Guide

[`CHANGELOG.md`](../CHANGELOG.md) at the repo root records notable changes to
this project. It follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Adding an entry

Add your entry under `## [Unreleased]` at the top of the file, in the section
that fits.

- **Added** for new features.
- **Changed** for changes in existing functionality.
- **Deprecated** for soon-to-be removed features.
- **Removed** for now removed features.
- **Fixed** for any bug fixes.

Add one entry per PR, in the section that fits best.
End with your GitHub handle and a link to the PR:
`@<github-handle> [#<PR number>](<PR URL>)`.

Example:

```markdown
## [Unreleased]

### Fixed

- Fixed the conda environment lock files so they work for everyone. @mavaylon1 [#123](https://github.com/slac-lcls/lcls2/pull/123)
```

## Cutting a release

1. Rename `## [Unreleased]` to the version and date, e.g. `## [1.2.0] - 2026-10-23`.
2. Delete any empty headings from that release.
3. Add a new `## [Unreleased]` section above it with all five headings, empty.
