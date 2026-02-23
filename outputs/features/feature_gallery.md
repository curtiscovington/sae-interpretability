# Feature Gallery (Top 10) — SAE Interpretability Experiment

This section is curated for the blog post: features with the clearest semantics from the latest run.

## A. Wiki-trained SAE (general text)

### 1) Feature 600 — **Pageant / biography event language**
- **Heuristic label:** `miss, universe, 2006`
- **Why it’s interesting:** Clusters around biography-style passages about pageants, years, and career summaries.
- **Example contexts:**
  - "represented her country at the 2006 world Miss Universe pageant..."
  - "represented Sri Lanka at the world Miss Universe 2006 pageant..."

### 2) Feature 23 — **Church/art-location entity cluster**
- **Heuristic label:** `croydon, andrew`
- **Why it’s interesting:** Fires on recurring named entities tied to place + institution in historical prose.
- **Example contexts:**
  - "at St. Andrew's, South Croydon..."
  - "in St Andrew's Church, South Croydon..."

### 3) Feature 200 — **Institutional/reporting prose**
- **Heuristic label:** `international, survey`
- **Why it’s interesting:** Picks up formal register around organizations and official bodies.
- **Example contexts:**
  - "International Civil Aviation Organization..."
  - "Archaeological Survey of India..."

### 4) Feature 747 — **Temporal narrative transitions**
- **Heuristic label:** `years, only`
- **Why it’s interesting:** Activates on chronology-heavy transitions and historical framing.
- **Example contexts:**
  - "For many years..."
  - "In its early years..."

### 5) Feature 138 — **Evidence/explanation connective prose**
- **Heuristic label:** `that, based`
- **Why it’s interesting:** Captures argumentative/explanatory clauses in encyclopedic text.
- **Example contexts:**
  - "Some scholars have argued, based in part on..."
  - "analysis based on mitochondrial and nuclear DNA..."

---

## B. Code-trained SAE (Python/code text)

### 6) Feature 479 — **Import / framework exception plumbing**
- **Heuristic label:** `import, from, werkzeug.exceptions`
- **Why it’s interesting:** Strongly code-specific, tied to framework-level import/error handling patterns.
- **Example contexts:**
  - Flask request/exception handling snippets
  - Werkzeug routing + response error paths

### 7) Feature 611 — **Control-flow with null/error exits**
- **Heuristic label:** `none, raise, return`
- **Why it’s interesting:** Captures canonical Python defensive flow: `None` checks, `raise`, then `return`.
- **Example contexts:**
  - encoding validation paths
  - API argument coercion and early-return guards

### 8) Feature 329 — **Tokenizer/file decoding error paths**
- **Heuristic label:** `error, return, filename`
- **Why it’s interesting:** Very specific to parsing/tokenization pipelines where filename + encoding are central.
- **Example contexts:**
  - `error(message, filename=...)`
  - `return encoding` / `read_or_stop()` branches

### 9) Feature 17 — **Branch-heavy AST/response handling**
- **Heuristic label:** `import, elif, error`
- **Why it’s interesting:** Mixes `elif`-dense AST transforms with web response fallback logic.
- **Example contexts:**
  - `elif isinstance(value, AST): ...`
  - debug-mode routing error replacement logic

### 10) Feature 637 — **Boolean comparator/recursion branches**
- **Heuristic label:** `else, return, encoding`
- **Why it’s interesting:** Captures repeated nested compare/return structures common in parser internals.
- **Example contexts:**
  - repeated `else: return ...` recursion checks
  - structural equality helper functions

---

## Short takeaways for the post

- **Code SAE features are clearer and more local** (imports, error handling, control flow, parser internals).
- **Wiki SAE features are more entity/genre-driven** (named entities, institutional prose, chronology).
- Even after stronger sparsity, many features remain fairly broad; next step is to push selectivity further (higher L1 or top-k SAE).
