# Contributing

Contributions should preserve the distinction between manuscript-specified
choices and implementation defaults.

1. Open an issue describing the paper construct, bug, or experiment affected.
2. Create a focused branch and add or update deterministic tests.
3. Run `make test` and `make lint`.
4. Do not commit model weights, TaxAI source, private prompts, generated logs,
   credentials, or participant data.
5. In a pull request, report the exact config, seed, hardware, and whether a
   result is a smoke test, reimplementation, or numerical reproduction.

Changes to reported reference values require a corresponding manuscript source
and must never be copied into generated-result folders.
