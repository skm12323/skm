import numpy as np
import tensorcircuit as tc

c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)

print(np.round(c.expectation_ps(z=[0, 1]), 6))