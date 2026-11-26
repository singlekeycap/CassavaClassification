from .imbalance import class_counts, class_weights, oversample
from .splits import describe, get_splits

tr, va, te = get_splits()
print(describe(tr, va, te))
print("\ninverse-freq weights (power=1):  ", class_weights(tr.label, 1.0).numpy().round(3))
print("inverse-freq weights (power=0.5):", class_weights(tr.label, 0.5).numpy().round(3))
print("oversample('median') counts:     ", class_counts(oversample(tr, "median").label))
