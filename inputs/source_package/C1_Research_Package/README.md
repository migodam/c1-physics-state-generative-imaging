# C1 Research Package

Date: 2026-10-13. Independent from B1 (historical A32 filenames).

Read `C1_THEORY_AND_ARCHITECTURE.md` for the research verdict, mathematical model, architectures and proof sketches. Read `C1_LITERATURE_AUDIT.md` for prior art and full-text access boundaries. `C1_CODEX_FIRST.md` and `C1_PROTOCOL.json` define the small E0/E1 implementation and conditional later gates.

Run the twelve non-training checks with:

```sh
python c1_algebra_checks.py --out C1_CHECKS.json
```

Requires NumPy and SciPy. The script uses no network, GPU, Maxwell solver or user files. These algebra checks are not evidence of successful learned imaging.

No C1 model was trained this turn. No Maxwell evaluations were performed. All E0-E5 model experiments are specifications, not results. The current strongest recommendation is a branch-contrast recurrent conditional generator, with an MDN/direct conditional-flow baseline and optional density-corrected full-wave weights.

The chat response contains the Chinese research report; the files use English for direct implementation handoff. No upstream paper PDFs, private credentials, or user background archives are redistributed.
