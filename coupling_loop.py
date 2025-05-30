import os
import math
import numpy as np
import itertools




# ——— USER CONFIGURATION ———

name_base = "coupling_loop_5_4p"

# 1) List every parameter your card supports
all_params = ['c0', 'c1', 'c2', 'c3', 'c4']

# 2) Declare which ones to hold fixed (and their values)
fixed_values = {
    # Example:
    # 'c0': -1560,
    # 'c1': -1560,
    # 'c2': -1560,
    # 'c3': -1560,
    # 'c4': -1560,
}

steps = 8

# 3) Per-parameter value ranges (int steps)
param_ranges = {
    'c0': np.linspace(-1560, 1560, steps, dtype=int),
    'c1': np.linspace(-1560, 1560, steps, dtype=int),
    'c2': np.linspace(-1560, 1560, steps, dtype=int),
    'c3': np.linspace(-1560, 1560, steps, dtype=int),
    'c4': np.linspace(-1560, 1560, steps, dtype=int),

    # 'c0': np.linspace(-936, -1560, steps, dtype=int),
    # 'c1': np.linspace(-936, -312, steps, dtype=int),
    # 'c2': np.linspace(-936, -312, steps, dtype=int),
    # 'c3': np.linspace(-312, 312, steps, dtype=int),
    # 'c4': np.linspace(0, 624, steps, dtype=int),
}

# 4) Parallel split (optional)
parallel = True
fragments = 10

# ——— END CONFIGURATION ———

# Determine which parameters to vary
vary_params = [p for p in all_params if p not in fixed_values]
vary_values = [param_ranges[p] for p in vary_params]


# input_card = """10090000300000000000 {name}
# +140.    +01.    +0.5
# +40+04+00+04+00+04
# +00.1   +20.
# +16.0   +1.0078 +01.    +150.00 +60.    +1.226          +00.85  +01.
# +01+01
# +01.    -56.709 +1.228  +0.656          -1.030  +1.228  +00.656 
# +02.                                    +38.332 +1.258  +00.597
# -04.    -23.308 +01.064 +0.590
# +00.000 +1.0078 +01.    +150.00 +60.    +01.30          +00.85  +01.    -0.130
# +02+02
# +01.    -56.709 +1.228  +0.656          -1.030  +1.228  +00.656 
# +02.                                    +38.332 +1.258  +00.597
# -04.    -23.308 +01.064 +0.590
# -03.932 +3.0160 +01.    +148.00 +60.    +01.30          +00.25  +01.    -0.000
# +03+03
# +01.    -161.738+1.200  +0.720          -21.207 +1.400  +00.840
# -04     -10.000 +1.200  +0.720
# -03.932 +3.0160 +01.    +148.00 +60.    +01.30          +00.25  +01.    -0.130
# +04+04
# +01.    -161.738+1.200  +0.720          -21.207 +1.400  +00.840
# -04     -10.000 +1.200  +0.720
# -02-01+02+00+04+00+03+01+0.284                  +01.25
# +02.    -56.709 +1.228  +0.656          -1.030  +1.228  +00.656 
# +03.                                    +38.332 +1.258  +00.597
# -05.    -23.308 +01.064 +0.590
# -04-03+02+00+04+00+03+01+0.201                  +01.30
# +02.    -161.738+1.200  +0.720          -21.207 +1.400  +00.840
# -05     -10.000 +1.200  +0.720
# -03-01+00+00+00+02+00+00{c0}
# +01.    +05.    +00.    +00.64
# -07.332 +01.008 +00.    +148.00 +60.                                    +0.000
# -01.    -01.    +01.17  +00.75  +25.
# +01.    +03.    +07.    +01.    +60.
# +00
# -04-01+02+00+04+02+00+00{c1}
# +01.    +05.    +00.    +01.00
# -07.332 +01.008 +00.    +148.00 +60.                                    +0.150
# -01.    -01.    +01.17  +00.75  +25.
# +01.    +03.    +07.    +01.    +60.
# +00
# -04-02+00+00+00+02+00+00{c2}
# +01.    +05.    +00.    +00.66
# -07.332 +01.008 +00.    +148.00 +60.                                    +0.150
# -01.    -01.    +01.17  +00.75  +25.
# +01.    +03.    +07.    +01.    +60.
# +00
# -04-02+02+00+04+02+00+00{c3}
# +01.    +05.    +00.    +00.66
# -07.332 +01.008 +00.    +148.00 +60.                                    +0.150
# -01.    -01.    +01.17  +00.75  +25.
# +01.    +03.    +07.    +01.    +60.
# +00
# -04-02+04+00+08+02+00+00{c4}
# +01.    +05.    +00.    +00.66
# -07.332 +01.008 +00.    +148.00 +60.                                    +0.150
# -01.    -01.    +01.17  +00.75  +25.
# +01.    +03.    +07.    +01.    +60.
# +00
# +00+00
# """

input_card = """10090000300000000000 {name}
+140.    +01.    +0.5
+40+04+00+08+00+04
+00.1   +20.
+16.0   +1.0078 +01.    +150.00 +60.    +1.226          +00.85  +01.
+01+01
+01.    -56.709 +1.228  +0.656          -1.030  +1.228  +00.656 
+02.                                    +38.332 +1.258  +00.597
-04.    -23.308 +01.064 +0.590
+00.000 +1.0078 +01.    +150.00 +60.    +01.30          +00.85  +01.    -0.381
+02+02
+01.    -56.709 +1.228  +0.656          -1.030  +1.228  +00.656 
+02.                                    +38.332 +1.258  +00.597
-04.    -23.308 +01.064 +0.590
-03.932 +3.0160 +01.    +148.00 +60.    +01.30          +00.25  +01.    -0.000
+03+03
+01.    -161.738+1.200  +0.720          -21.207 +1.400  +00.840
-04     -10.000 +1.200  +0.720
-03.932 +3.0160 +01.    +148.00 +60.    +01.30          +00.25  +01.    -0.150
+04+04
+01.    -161.738+1.200  +0.720          -21.207 +1.400  +00.840
-04     -10.000 +1.200  +0.720
-02-01+04+00+08+00+03+01+0.284                  +01.25
+02.    -56.709 +1.228  +0.656          -1.030  +1.228  +00.656 
+03.                                    +38.332 +1.258  +00.597
-05.    -23.308 +01.064 +0.590
-04-03+02+00+04+00+03+01+0.201                  +01.30
+02.    -161.738+1.200  +0.720          -21.207 +1.400  +00.840
-05     -10.000 +1.200  +0.720
-03-01+00+00+00+02+00+00{c0}
+01.    +05.    +00.    +00.64
-07.332 +01.008 +00.    +148.00 +60.                                    +0.000
-01.    -01.    +01.17  +00.75  +25.
+01.    +03.    +07.    +01.    +60.
+00
-04-01+02+00+04+02+00+00{c1}
+01.    +05.    +00.    +01.00
-07.332 +01.008 +00.    +148.00 +60.                                    +0.150
-01.    -01.    +01.17  +00.75  +25.
+01.    +03.    +07.    +01.    +60.
+00
-04-02+02+00+04+02+00+00{c2}
+01.    +05.    +00.    +00.66
-07.332 +01.008 +00.    +148.00 +60.                                    +0.150
-01.    -01.    +01.17  +00.75  +25.
+01.    +03.    +07.    +01.    +60.
+00
-04-02+04+00+08+02+00+00{c3}
+01.    +05.    +00.    +00.66
-07.332 +01.008 +00.    +148.00 +60.                                    +0.150
-01.    -01.    +01.17  +00.75  +25.
+01.    +03.    +07.    +01.    +60.
+00
-04-02+06+00+12+02+00+00{c4}
+01.    +05.    +00.    +00.66
-07.332 +01.008 +00.    +148.00 +60.                                    +0.150
-01.    -01.    +01.17  +00.75  +25.
+01.    +03.    +07.    +01.    +60.
+00
+00+00
"""


# Generate all combinations
cards = []
for combo in itertools.product(*vary_values):
    params = dict(fixed_values)
    params.update({p: float(v) for p, v in zip(vary_params, combo)})

    parts = [f"{p}{int(params[p]):+05d}" for p in all_params]
    unique_name = "_".join(parts)

    cards.append(input_card.format(
        name=unique_name,
        **{p: f"{params.get(p, 0):+08.2f}" for p in all_params}
    ))

total = len(cards)
print(f"Will write {total} cards (varying {vary_params})")

if parallel:
    os.makedirs(name_base, exist_ok=True)
    chunk = math.ceil(total / fragments)
    for i in range(fragments):
        frag = cards[i*chunk:(i+1)*chunk]
        if not frag:
            break
        path = f"{name_base}/{name_base}_part{i+1}.in"
        with open(path, "w") as f:
            f.write("".join(frag) + "9")
        print(f"Wrote fragment {i+1} ({len(frag)} cards) → {path}")
else:
    out = f"{name_base}.in"
    with open(out, "w") as f:
        f.write("".join(cards) + "9")
    print(f"Wrote single file → {out}")