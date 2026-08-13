# sample_data

Placeholder directory. No real building photos are included in this
repository (and none should be fabricated or downloaded into it).

To try `facade_parser_cli` on a real photo, supply your own already
rectified (orthorectified, perspective-corrected) facade image and point
the CLI at it:

```
./facade_parser_cli --input /path/to/your_facade.png --output ./out
```

The automated test suite does not depend on anything in this directory —
it uses procedurally generated synthetic facades (see
`tests/synthetic_facade.hpp`) so it stays deterministic and
ground-truth-backed without real photos. See `docs/PLAN.md` for details.
