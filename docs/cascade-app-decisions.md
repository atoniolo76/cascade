# Cascade app decisions

Owner: Amani. Scope: the Cascade app container (main mode + simulator mode shell), its API, and its UI. See [charter.md](charter.md).

Decisions are recorded as they are made. Superseded decisions are struck through with a note.

## Process

- Work happens on the `amani` branch, committed continuously.

## Stack

- Backend: Python FastAPI. Frontend: React built with Vite into static files that FastAPI serves.
- One process, one image for both main and simulator modes. Next.js was rejected because it needs a Node server next to Python in every forked sailbox, and its SSR/API-route strengths don't apply to a Python-backed dashboard.

## Simulator → main communication

- Push over plain HTTP. Each simulator gets the main app's URL and its own sim id via env vars at fork time.
- Simulators call `POST /api/sims/{id}/heartbeat` (health/status) and `POST /api/sims/{id}/logs` (log chunks). This is the contract with Alessio's simulator side.
- Main fans updates out to the browser over SSE.
- Simulators need no public endpoint of their own.
