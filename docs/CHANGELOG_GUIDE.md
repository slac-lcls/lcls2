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

Write entries for users, not as a list of commits: one short line saying what
changed. Link the PR or issue if it helps, e.g. `(#123)`.

Example:

```markdown
## [Unreleased]

### Fixed

- Updated the conda environment lock files so they work for everyone. (#123).
```

## Cutting a release

1. Rename `## [Unreleased]` to the version and date, e.g. `## [1.2.0] - 2026-10-23`.
2. Delete any empty headings from that release.
3. Add a new `## [Unreleased]` section above it with all five headings, empty.
