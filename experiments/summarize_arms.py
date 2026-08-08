"""Summarize the three split arms — written before the run finished, deliberately.

The comparison and the decision rule are fixed here while the `official` numbers are still
unknown, so the conclusion can't be reverse-engineered from whichever result arrives. The
prediction on record: official inflation lands between clean (~-0.8pt) and leaky (~+3.3pt).

Reads reports/leakage_rigorous.json. Prints a table and states which of three outcomes holds:

  A. official ≈ clean      -> the shipped split is sufficient in practice; the advice simplifies
                              to "use dish_ids/splits" and session grouping is a refinement
  B. official ≈ leaky      -> the shipped split does not protect selection; session grouping is
                              necessary, and the 34.7% contamination is doing real damage
  C. official in between   -> partial protection; report the fraction of the gap it closes

    python experiments/summarize_arms.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPORT = Path("reports/leakage_rigorous.json")
ARMS = ("leaky", "official", "clean")


def main() -> None:
    cells = json.loads(REPORT.read_text())["cells"]
    by_arm = {a: [c for c in cells if c["strategy"] == a] for a in ARMS}

    print(f"{'arm':<10} {'n':>3}  {'val_acc':>8} {'test_acc':>9} {'inflation':>10}  positive")
    print("-" * 56)
    means: dict[str, float] = {}
    for arm in ARMS:
        g = by_arm[arm]
        if not g:
            print(f"{arm:<10} {'—':>3}  (not run yet)")
            continue
        infl = [c["inflation"] for c in g]
        means[arm] = 100 * st.mean(infl)
        print(
            f"{arm:<10} {len(g):>3}  {st.mean(c['val_acc'] for c in g):>8.3f} "
            f"{st.mean(c['test_acc'] for c in g):>9.3f} {means[arm]:>+9.1f}pt  "
            f"{sum(x > 0 for x in infl)}/{len(infl)}"
        )

    if len(means) < 3:
        print("\nincomplete — rerun once all 30 cells are in")
        return

    leaky, clean, official = means["leaky"], means["clean"], means["official"]
    span = leaky - clean
    closed = (leaky - official) / span if span else float("nan")

    print(f"\ngap between leaky and clean: {span:.1f}pt")
    print(f"official closes {100 * closed:.0f}% of it (inflation {official:+.1f}pt)")

    # Decision rule fixed in advance: within 25% of an endpoint counts as "resembles" it.
    if closed >= 0.75:
        verdict = "A — official ≈ clean: the shipped split largely protects selection"
    elif closed <= 0.25:
        verdict = "B — official ≈ leaky: the shipped split does NOT protect selection"
    else:
        verdict = "C — partial: the shipped split helps but leaves real inflation"
    print(f"\nverdict: {verdict}")
    print(
        "\nCaveat to carry into any write-up: official val is ~15.5% of the pool against 20% "
        "for the other arms, so some of any difference is selection noise from a smaller "
        "validation set, not leakage. Per-cell n_val is in the report."
    )


if __name__ == "__main__":
    main()
