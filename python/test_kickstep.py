"""python test_kickstep.py: the NumPy reference matches the C# streaming inference (fixture: random weights, random input)."""
import os

import numpy as np

import kickstep

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixture")
model = kickstep.load(os.path.join(here, "fixture.thdm"))
x = np.fromfile(os.path.join(here, "fixture.in.f32"), "<f4").reshape(-1, kickstep.BANDS)
want = np.fromfile(os.path.join(here, "fixture.out.f32"), "<f4").reshape(-1, 3)
worst = np.abs(kickstep.probabilities(model, x) - want).max()
assert worst < 1e-4, worst
print(f"ok, worst difference {worst:.1e}")
