# Dionysus arXiv Tunnel Deployment Design

## Goal

Move the personal arXiv MCP runtime and its paper library from the laptop to
Dionysus so the existing OpenAI Secure MCP Tunnel remains available while the
laptop is asleep or offline.

## Observed state

- Dionysus is Ubuntu 24.04 x86_64, reachable through Tailscale, with 81 GB free.
- Its user systemd manager is running and lingering is enabled for `rookslog`.
- Dionysus does not yet have `uv`, `tunnel-client`, arXiv MCP configuration, or
  arXiv data.
- The laptop tunnel is currently stopped.
- The laptop library occupies 11 MB across 139 files under
  `~/.local/share/arxiv-mcp-server/papers`.
- The laptop's uv tool is the obsolete upstream `arxiv-mcp-server` 0.5.0.

## Approved design

Run the official Linux AMD64 `tunnel-client` and the current arXiv MCP branch as
one user-systemd service on Dionysus. The tunnel profile will keep the existing
tunnel ID and launch `arxiv-mcp-pro` over stdio with storage at
`~/.local/share/arxiv-mcp-pro/papers`.

Install the arXiv server from the exact approved commit (`935e4d4`) with the
`pdf` and lightweight `pro` extras. Pin the tunnel client to the official v0.0.9
Linux AMD64 release and verify it against the publisher's checksum file.

Store the runtime API key in a mode-0600 environment file on Dionysus. Transfer
it directly from macOS Keychain to the remote file without printing it. Bind the
tunnel health/UI listener to remote loopback only; operators can reach it over
an SSH port forward when needed.

Copy persistent paper Markdown and watch state with `rsync`. Do not copy the
transient arXiv lock or cooldown files. Compare source and destination manifests
before cutover. Keep the laptop copy and configuration intact as rollback until
the remote service passes health, readiness, MCP initialization, and tool-list
checks.

## Failure and rollback boundaries

- Do not start the remote service until installation, configuration, secret
  provisioning, and data verification have succeeded.
- Never run laptop and Dionysus tunnel clients against the same tunnel ID at the
  same time.
- If remote readiness fails, stop/disable the remote user service and retain the
  existing laptop launcher unchanged.
- Local data deletion and final retirement of the laptop runtime are explicitly
  outside this migration.

## Verification

- Verify downloaded tunnel-client checksum and version.
- Verify installed `arxiv-mcp-pro` package version, source identity, and backend.
- Verify source/destination persistent-data manifests.
- Run `tunnel-client doctor --explain` on Dionysus.
- Verify systemd active/enabled state and `/healthz` plus `/readyz`.
- Exercise MCP initialize and tools/list through the tunnel client or its doctor
  diagnostics without invoking external arXiv requests.

