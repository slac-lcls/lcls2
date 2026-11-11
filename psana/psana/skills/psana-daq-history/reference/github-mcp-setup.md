# Setting up a GitHub MCP server for issue filing

`psana-daq-history` files GitHub issues/comments via an MCP tool
(`issue_write`, `add_issue_comment`) when one is configured in the current
agent session. This doc covers the one-time, per-user setup for three
harnesses: opencode, Claude Code, and Codex. If you're on a different
harness, the token/scope guidance below still applies — consult your
harness's own MCP documentation for the config syntax.

If you'd rather not set this up right now, that's fine: `psana-daq-history`
falls back to showing you a fully drafted issue/comment to paste into
GitHub yourself. This doc is only needed to make filing happen directly
from the agent.

---

## 1. Create a fine-grained personal access token

At https://github.com/settings/personal-access-tokens/new:

- **Set "Resource owner" to the target org (e.g. `slac-lcls`) before
  anything else.** It defaults to your own account, and while it's there
  the repository picker lists only repos you personally own — the target
  repo never appears, making it look like you lack access. This is the
  single most common setup failure.
- Repository access → **Only select repositories** → the target repo
  (e.g. `slac-lcls/lcls2`).
- Permissions → Repository permissions → **Issues: Read and write**.
  Leave everything else at *No access*. `Metadata: Read-only` is added
  automatically and is mandatory.

**If the org isn't offered as a resource owner**, it hasn't enabled
fine-grained tokens; an org owner can allow this under Organization
Settings → Third-party Access → Personal access tokens. A classic token
works as a fallback, but note it has **no issues-only scope** —
`public_repo` is the narrowest that can file an issue, and it also grants
code-push on every public repo you can write to. Prefer fine-grained.
Don't use `repo`: it adds private-repo access this skill never needs.

**Tokens are personal — never shared between people.** Issues and
comments filed via MCP carry the invoking user's own GitHub identity. If
a shared identity is ever genuinely wanted, the supported pattern is a
dedicated machine account owning its own token, not a circulated personal
one.

## 2. Store the token where only you can read it

```bash
mkdir -p ~/.config/daq-diagnostics
printf '%s\n' '<token>' > ~/.config/daq-diagnostics/github_token
chmod 600 ~/.config/daq-diagnostics/github_token
```

Mode `600` matters — this token can write to a shared repo. Avoid trailing
whitespace: depending on how your harness substitutes the file's contents
into an `Authorization` header, a stray character can make it malformed.

This path is shared across harnesses deliberately, so switching agents
doesn't mean recreating the token.

## 3. Configure your harness's GitHub MCP server

All three harnesses below point at the same endpoint:
`https://api.githubcopilot.com/mcp/x/issues`. The `/x/issues` suffix loads
**only** the issues toolset — plain `/mcp/` also works but additionally
loads repos, PRs, actions, and code scanning, which is context cost on
every session for capability this skill never uses.

### opencode

Add a `github` entry to your own `~/.config/opencode/opencode.jsonc`
(per-user — not centrally provisioned). It goes *alongside* any MCP
servers you already have, inside the existing `"mcp"` block — don't
replace the block, and remember the comma after the preceding entry:

```jsonc
"mcp": {
  // ...existing entries, each followed by a comma...
  "github": {
    "type": "remote",
    "url": "https://api.githubcopilot.com/mcp/x/issues",
    "enabled": true,
    "headers": {
      "Authorization": "Bearer {file:~/.config/daq-diagnostics/github_token}"
    }
  }
}
```

(`verified-live`, 2026-09-29: with `/x/issues` the session exposed issue,
label, and sub-issue tools and no repo, PR, or actions tools, and
successfully authenticated a real filing against `slac-lcls/lcls2`.)

**Restart the opencode session.** MCP servers connect at startup, so a
running session won't pick up a new entry.

### Claude Code

```bash
claude mcp add --transport http github \
  https://api.githubcopilot.com/mcp/x/issues \
  --header "Authorization: Bearer $(cat ~/.config/daq-diagnostics/github_token)"
```

This embeds the token's current value as a static header at the time you
run the command (per Claude Code's MCP docs, `claude mcp add --transport
http ... --header`) — if you rotate the token, re-run this command rather
than expecting it to re-read the file. Use `--scope user` instead of the
default local scope if you want this available across every project, not
just the one you're in when you run it.

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.github]
url = "https://api.githubcopilot.com/mcp/x/issues"
bearer_token_env_var = "GITHUB_MCP_TOKEN"
```

Then export the token in the shell Codex starts from (e.g. your shell
profile, so it's set before Codex launches):

```bash
export GITHUB_MCP_TOKEN="$(cat ~/.config/daq-diagnostics/github_token)"
```

## 4. Verify, without writing anything

Request a label on the repo (e.g. `get_label` on `bug`, which exists by
default on every repo). This exercises the token end to end and creates
nothing. Note that seeing GitHub-related tools in a session proves only
that the *server* connected — the tool manifest loads before any
authenticated call, so a bad token still shows tools and fails only at
call time.

If a known-good token is rejected by `api.githubcopilot.com`, an active
Copilot entitlement may be required.

## Known gaps

**Not wired up for AMI-spawned agents today.** AMI's `mcp_server.py`
registers only its own `ami` MCP server for agents it spawns — a
personally configured GitHub MCP server does not extend to a session
spawned from inside AMI. The manual-paste fallback in
`psana-daq-history/SKILL.md` works in that context too; this is a known
gap for agent-side filing specifically from within an AMI-spawned
session, not a blocker.

---

## Label API mechanics

`psana-daq-history/SKILL.md`'s "Labels" section covers the policy (query
real labels, suggest, let the human pick, never create silently). This
section covers the mechanics behind that policy, for anyone extending or
debugging it.

**Why `POST /labels` happens before `issue_write`, not as a side effect of
it:** passing a nonexistent label name in `issue_write`'s `labels` array
does create the label as a side effect — but with no `color`/`description`
chosen, so GitHub defaults it to flat grey (`#ededed`). An explicit `POST
/repos/slac-lcls/lcls2/labels` first, with a chosen `color` and
`description`, gives the label a real appearance before `issue_write` ever
references it — so it reads as a deliberate part of the vocabulary rather
than an accident of API ordering. Verified on all three labels created for
issue #134 (`daq-diagnostics`, `hutch-mfx`, `component-epix`): all three
would have landed grey if created implicitly, and didn't because the
explicit `POST` ran first.

**Issue #134 finding, concretely:** before the first real filing
(2026-09-29), the repo had only the 7 GitHub-default labels (`bug`,
`duplicate`, `enhancement`, `help wanted`, `invalid`, `question`,
`wontfix`), no custom labels, and no labeled issues. Filing issue #134
created `daq-diagnostics`, `hutch-mfx`, and `component-epix` as a
consequence of the human confirming the suggested labels at that point.
This list only grows — re-query `GET /repos/slac-lcls/lcls2/labels` rather
than trusting either this snapshot or any later one as current.

**Org-level custom issue fields exist and are explicitly not used here:**
`slac-lcls` defines org-level custom issue fields (`Priority`, `Start
date`, `Target date`, `Effort` — all `ORG_ONLY` visibility, as of
2026-09-29). `psana-daq-history` does not set any of them; labels remain
the only metadata it writes. Worth knowing if a future extension
considers using these fields instead of (or alongside) labels.

---

## Slack-thread rationale and rejected alternatives

`#lcls-daq-diagnostics` is subscribed to `slac-lcls/lcls2`'s issue
activity (one-time, human-performed setup, done 2026-09-29) —
**GitHub's native Slack app auto-creates a thread the moment an issue is
filed.** No bot token, no `chat:write` scope, and no code in this skill
are involved — it's entirely GitHub's own integration, external to
anything this suite does. (`verified-live`, 2026-09-29: filing issue #134
created a thread in that channel with no action from this skill beyond
the filing itself.)

**`psana-daq-history` never reads that thread, by design — this was
considered and rejected during planning**, not simply left undone:
Slack's `search.messages` API requires a user token (impersonating a
specific human, with access to whatever that human can see, including
private channels), not a bot token — a materially larger trust decision
than this suite makes anywhere else, for a capability (reading back
informal chat) that's reachable more simply by searching the issue
itself.

**The mitigation, stated as a convention, not enforced by any tool:** if
discussion in the thread reaches a conclusion — a cause identified, a fix
that worked, a hypothesis ruled out — **post it back to the issue as a
comment.** The issue, not the thread, is what a later `psana-daq` sweep's
search actually finds. A conclusion that lives only in Slack is invisible
to every future sweep.
