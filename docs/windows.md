# Windows

- `run-demo.cmd` runs the CLI demo.
- `run-api.cmd` starts the pre-indexed demo FastAPI app.

Both wrappers use a process-local PowerShell execution-policy bypass and project-local `.run/tmp`. They do not modify the machine-wide execution policy.
