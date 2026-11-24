# Midas — Privacy Policy

_Last updated: 2026-11-24_

Midas is a **local-first** memory connector. It is designed so that your data never leaves your own
computer. This policy describes exactly what Midas does and does not do with your data.

## What data Midas handles

Midas stores the text you (or Claude on your behalf) choose to remember — facts, decisions,
preferences, constraints, and conversation turns — together with derived metadata (a vector
embedding, an importance score, timestamps, a `kind` label, and a session id).

## Where it is stored

All memories are stored **locally on your machine** in a single SQLite file that you control
(default: `~/.midas/memory.sqlite3`, configurable). Midas does not operate any hosted server, account, or cloud
storage, and there is no telemetry or analytics.

## Data collection

The Midas developer collects **no data**. We do not receive, see, or have access to your memories,
prompts, embeddings, or usage. There is no sign-up and no API key.

## Network usage

- **No memory data is ever transmitted over the network.** Capture, recall, forgetting, and context
  assembly all run entirely offline using local computation.
- The only outbound network activity is **infrastructure, not your data**:
  - One-time download of the open-source embedding model (BGE ONNX, from Hugging Face) the first
    time the `local` embedding backend is used. Select the `hashing` backend to avoid this entirely.
  - Package installation and user-initiated updates through a package manager.
  - A request to PyPI for public Midas version metadata. Automatic checks are cached for 24 hours
    and can be disabled with `MIDAS_UPDATE_CHECK=0`. The request includes no memories, prompts,
    database paths, or client identifiers.

## Third-party sharing

None. Midas does not share data with any third party because it never collects or transmits your
data in the first place.

## Data retention and deletion

- You control retention. Memories persist in your local SQLite file until you remove them.
- Midas can bound storage automatically by forgetting the lowest-value memories (no LLM involved).
- You can delete a single memory (`forget`), erase everything about a topic (`forget_matching` —
  dry-run preview first, and the deleted records are returned as an audit trail), run a
  retention/audit pass (`maintain`), or erase everything (`forget_all`). Deleting the SQLite file
  removes all stored memories permanently.

## Contact

Questions or security reports: open an issue at
<https://github.com/vornicx/Midas/issues> or contact the maintainer via the GitHub profile at
<https://github.com/vornicx>.
