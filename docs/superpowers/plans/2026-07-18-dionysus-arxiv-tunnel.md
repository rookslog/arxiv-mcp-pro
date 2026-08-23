# Dionysus arXiv Tunnel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the current arXiv MCP server and existing OpenAI tunnel as an always-on user service on Dionysus, with the laptop paper library migrated safely.

**Architecture:** A user-systemd unit runs the official `tunnel-client`, which launches `arxiv-mcp-pro` as its stdio MCP child. Configuration, the runtime secret, and persistent paper data live in separate mode-appropriate directories under the Dionysus user account.

**Tech Stack:** Ubuntu 24.04, systemd user services, OpenAI `tunnel-client` v0.0.9, uv, Python 3.12, rsync, Tailscale/SSH.

---

### Task 1: Install pinned runtimes

**Files:**
- Create: `/home/rookslog/.local/bin/tunnel-client`
- Create: `/home/rookslog/.local/bin/arxiv-mcp-pro` (uv-managed entry point)

- [x] Download the official `linux-amd64.zip` and `SHA256SUMS.txt` for tunnel-client v0.0.9 into a temporary directory on Dionysus.
- [x] Verify the archive with `sha256sum --check` before installing its binary.
- [x] Install pinned uv 0.11.28 into an isolated bootstrap venv (Dionysus already had an older uv binary outside its non-interactive SSH `PATH`).
- [x] Install `arxiv-mcp-pro[pdf,pro]` from git commit `935e4d4` into an isolated uv tool environment.
- [x] Confirm tunnel-client version `0.0.9` and arXiv package version `0.8.0`.

### Task 2: Stage configuration and service

**Files:**
- Create: `/home/rookslog/.config/tunnel-client/arxiv-local.yaml`
- Create: `/home/rookslog/.config/tunnel-client/arxiv-local.env`
- Create: `/home/rookslog/.config/systemd/user/arxiv-mcp-tunnel.service`

- [x] Write the profile with the existing tunnel ID, loopback health listener,
  and absolute `arxiv-mcp-pro` command using the new storage path.
- [x] Transfer `CONTROL_PLANE_API_KEY` directly from macOS Keychain into the
  remote environment file with mode 0600; do not print the value.
- [x] Write a user service with `Restart=always`, a bounded restart delay,
  `UMask=0077`, and the remote environment file.
- [x] Run `systemd-analyze --user verify` and `systemctl --user daemon-reload`
  without starting the service.

### Task 3: Copy and verify persistent application state

**Files:**
- Create: `/home/rookslog/.local/share/arxiv-mcp-pro/papers/*.md`
- Create: `/home/rookslog/.local/share/arxiv-mcp-pro/papers/watched_topics.json`

- [x] Identify the 137 persistent laptop files, excluding `arxiv_api.lock` and
  `arxiv_api.cooldown`.
- [x] Use `rsync --archive` with the same exclusions to copy the library.
- [x] Run a checksum-mode rsync dry run against the remote destination.
- [x] Confirm the checksum dry run reports no differences and both sides contain
  137 persistent files.

### Task 4: Validate and start the service

- [x] Run `tunnel-client doctor --profile arxiv-local --explain` with the remote
  environment file loaded.
- [x] Enable and start `arxiv-mcp-tunnel.service`.
- [x] Confirm `systemctl --user is-enabled` and `is-active` both succeed.
- [x] Confirm `tunnel-client health` reports both health and readiness.
- [x] Inspect bounded service logs for errors without emitting the API key.

### Task 5: Cutover verification and laptop polish

**Files:**
- Modify: `/Users/rookslog/.local/bin/run-arxiv-tunnel`
- Modify: `/Users/rookslog/.local/bin/check-arxiv-tunnel`

- [x] Verify the remote MCP child advertises the expected current tool surface.
- [x] Change the laptop run script into a remote `systemctl --user start` helper.
- [x] Change the laptop check script into a remote service/readiness probe.
- [x] Confirm both helpers work while preserving the laptop paper library and
  old local profile as rollback artifacts.
- [x] Record final commands, versions, manifest counts, health evidence, and
  rollback boundary in the task handoff.

## Execution record

- Deployed host: `DIONYSUS` (`rookslog` user systemd manager, lingering enabled).
- Tunnel client: `0.0.9+62b9b42f698ec5319d2115e0c0ff1dcf6557d7ae`,
  installed from the checksum-verified official Linux AMD64 archive.
- Server: `arxiv-mcp-pro` 0.8.0 installed from exact git revision `935e4d4`
  with `pdf` and `pro` extras; model2vec is present and sentence-transformers is
  absent.
- Persistent state: 137 files / 11 MB copied; checksum-mode rsync dry run
  reported no differences. Transient lock and cooldown files were excluded.
- Service: `arxiv-mcp-tunnel.service` is enabled and active with zero restarts
  and `ExecMainStatus=0` at final verification.
- Health: `/healthz` returned HTTP 200 `live`; `/readyz` returned HTTP 200
  `ready`; PID cross-check passed.
- MCP smoke check: initialization returned server `arxiv-mcp-pro` version 0.8.0
  and the expected 11 tools.
- Repository verification: 303 tests passed with 8 warnings.
- Rollback: laptop papers, old uv tool, and local tunnel profile remain intact.
  The laptop run/check scripts now manage and inspect the Dionysus service.
- Follow-up boundary: OpenAI tunnel metadata still describes the server as
  running on the Mac. Updating it requires an admin API key with tunnel-management
  authority and was not attempted with the runtime key.
- Known non-failing diagnostic: onnxruntime emits a GPU-device discovery warning
  on this headless host; model2vec loads and the MCP server initializes normally.
