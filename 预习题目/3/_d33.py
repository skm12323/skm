import sys, numpy as np, tensorcircuit as tc
K = tc.set_backend("jax")
if "--c128" in sys.argv:
    tc.set_dtype("complex128")
sx = K.stack([K.stack([0,1]),K.stack([1,0])])
sy = K.stack([K.stack([0,-1j]),K.stack([1j,0])])
def f(theta):
    e1 = K.expm(-1j*theta*sx/2); e2 = K.expm(1j*theta*sx/2)
    P = K.matmul(e1, K.matmul(sy, e2))
    v0 = K.stack([1,0])
    return float(K.numpy(K.tensordot(K.conj(v0), K.tensordot(P,v0,axes=1), axes=1)).real)
lab = "complex128" if "--c128" in sys.argv else "complex64"
print(f"[{lab}] f(0.57)={f(0.57):.8f}  (sin0.57={np.sin(0.57):.8f})")
h=1e-3
print(f"[{lab}] f(0.571)={f(0.57+h):.8f}")
print(f"[{lab}] forward grad={(f(0.57+h)-f(0.57))/h:.6f}  (cos0.57={np.cos(0.57):.6f})")
# sweep h to find breakdown
print(f"[{lab}] h-sweep forward grad:")
for h in [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]:
    g=(f(0.57+h)-f(0.57))/h
    print(f"    h={h:.0e}  grad={g:.6e}")
