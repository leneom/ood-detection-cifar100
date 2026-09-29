# Audited three-seed comparison

Colored points are the three independently trained classifiers, with identical data split and scoring protocol. Black diamonds and whiskers show mean and sample SD across training seeds; these are not confidence intervals and differ from the single-classifier sample bootstrap intervals in each seed report. Symmetric SD whiskers can extend outside the rate range; the plotted seed rates stay within 0–100%.

Synthetic diagnostics are labeled separately and excluded from the natural macro. All methods keep their original temperature, score direction and training-only feature fitting.

![Training seed comparison](seed-comparison.png)

Input summary SHA-256: `007980a702dc41a757b30e66a3896a14e60240574f909d01ee223a1cc8aeeb5b`.

The independent audit passed for the exact summary plotted here. Figures are descriptive and do not establish statistical significance or select a method from OOD tests.
