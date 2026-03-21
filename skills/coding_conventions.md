# GreenSpace CNN — Coding Conventions & Style Guide

> Transferable reference for any agent or collaborator working on this codebase.
> Last updated: 2026-03-21

---

## Guiding Principle

**Clean, readable code that is reusable and scalable — without accumulating technical debt.**

Every change should be easy to read, easy to hand off, and easy to extend. Don't introduce shortcuts, duplications, or inconsistencies that make future changes harder. When you touch code, leave it cleaner than you found it.

---

## 1. Prefer Built-In Over Custom

Use standard library / framework functions whenever they can do the job. Only write custom code when the built-in genuinely falls short.

| Situation | Preferred | Avoid |
|---|---|---|
| Weighted AUC metric | `tf.keras.metrics.AUC(label_weights=...)` | Custom callback with sklearn (unless toggle is on) |
| Label column derivation | Shared function in `src/label_schema.py` | Inline re-derivation in every cell |
| Timestamp tagging | `datetime.now().strftime(...)` | Custom timestamp utilities |

When both a built-in and custom path exist, keep both behind a **toggle** (e.g., `USE_BUILTIN_WEIGHTED_AUC = True`). Don't delete the alternative — the user evaluates trade-offs across runs.

---

## 2. Define Once, Reuse Everywhere

| Variable | Define in | Reuse in |
|---|---|---|
| `RUN_TAG` | First config cell (cell-1 or cell-4) | All downstream cells |
| `binary_cols`, `bin_names`, `hard_bin_cols` | Once via `resolve_label_cols(train_df)` | Cells 3, 4, 6, 7 |
| `from datetime import datetime` | Top imports cell | Nowhere else |
| `run_dir` | Cell where artifacts are first needed | All artifact-saving cells |

Pattern for `RUN_TAG`:
```python
RUN_TAG = globals().get('RUN_TAG', None) or datetime.now().strftime('%Y%m%d_%H%M%S')
```
`globals().get()` allows external scripts to inject a tag before running the notebook.

---

## 3. Extract Shared Logic to `src/`

When the same derivation appears in multiple notebooks, move it to `src/`:

```
src/
  label_schema.py      # resolve_label_cols(df) → binary_cols, bin_names, hard_bin_cols
```

Keep extracted functions **short** (5–15 lines). No class hierarchies. No abstract base classes.

---

## 4. Toggles, Not Config Files

A/B decisions live as booleans at the top of the relevant notebook cell:

```python
USE_AUGMENTATION = True
USE_GARDEN_OVERSAMPLING = True
USE_FOCAL_LOSS = False
USE_BUILTIN_WEIGHTED_AUC = True
```

Each toggle has a brief comment explaining what it controls. No external config files, no CLI argument parsing.

---

## 5. Warnings Over Complex Guardrails

For unlikely edge cases, use a lightweight warning — don't build an elaborate automatic fix.

Example: `SingleClassValWarning` fires once at `on_train_begin`, prints a message, and stops there. It doesn't try to rebalance the data or switch AUC modes automatically.

---

## 6. Comments: Why, Not What

Add a comment when:
- A design decision might surprise a future reader
- A variable is defined elsewhere and reused here

Skip comments when:
- The code is self-explanatory
- You'd just be restating the function name

```python
# Good
# RUN_TAG defined once in cell-4; reused here without re-definition.

# Good
# Shade loss masked by walking_paths — shade is meaningless when no paths exist.

# Bad — just restating the code
# This loop iterates over binary columns.
```

---

## 7. Notebook Cell Discipline

- Don't add or remove cells unless explicitly asked. Cells are a mental map.
- Edit within existing cells.
- Keep cell numbering stable across sessions.
- One cell = one logical unit (imports, data pipeline, model definition, training, saving).

---

## 8. No Over-Engineering

- No error handling for impossible scenarios.
- No abstractions for one-time operations.
- No type annotations or docstrings on unchanged code.
- Three similar lines > one premature abstraction.
- No feature flags or backward-compatibility shims — just change the code.

---

## Quick Reference: Streamlining Checklist

When reviewing a notebook for cleanup:

1. **Duplicate definitions?** → Consolidate to earliest cell, add comment at reuse sites.
2. **Same derivation in multiple notebooks?** → Extract to `src/`.
3. **Custom code where built-in works?** → Switch to built-in, keep custom behind toggle.
4. **Imports scattered across cells?** → Move to top cell.
5. **Dead code?** → Delete it. No `# removed` comments.
