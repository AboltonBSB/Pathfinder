# pathFinder

Experimental tools for analyzing **Path of Exile** passive skill trees using spectral graph methods — Laplacian eigenvectors, Fiedler vectors, and clustering over tree structure and node weights (keystones, notables, etc.).

> **Early development:** This project is in a very early stage. Core pieces exist (Rust compute, a FastAPI backend, basic visualization), but it is not feature-complete, documented end-to-end, or ready for general use. Expect rough edges, incomplete workflows, and breaking changes.

## What it does (so far)

- Builds graph representations of skill tree nodes grouped by ascendancy (or the main tree)
- Computes Laplacian eigenvectors in **Rust** (via PyO3) for spectral analysis
- Runs K-means clustering on spectral embeddings
- Exposes compute/store/fetch endpoints through a **FastAPI** server
- Includes Python helpers for plotting Fiedler vectors, embeddings, and clusters

## Tech stack

| Layer | Tools |
|-------|--------|
| Compute | Rust, PyO3, faer, linfa, petgraph |
| API | Python, FastAPI, SQLAlchemy |
| Visualization | Python (matplotlib and related helpers) |

## Project layout

```
src/
  lib.rs              # Rust extension (Laplacian + clustering)
  api/                # FastAPI routes and models
  db/                 # Database setup
  helpers/            # Graph building, visualization, diagnostics
assets/
  data.json           # Skill tree node data
```

## Getting started (local)

Prerequisites: **Rust**, **Python 3.8+**, and **maturin**.

1. Build and install the Rust extension into your Python environment:

   ```bash
   maturin develop
   ```

2. Install Python dependencies (FastAPI, uvicorn, SQLAlchemy, requests, scikit-learn, etc.) — a formal dependency list is not set up yet.

3. From the `src/` directory, start the API server:

   ```bash
   cd src
   python -m api.routes
   ```

   The server runs at `http://localhost:8000`.

4. With the server running, run the client script (also from `src/`):

   ```bash
   python __main__.py
   ```

## Status

This is a personal/experimental project. There is no stable API, release process, or test suite yet. Contributions and feedback are welcome, but treat everything as work-in-progress.
