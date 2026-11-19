"""Compare every PNG in Images/plots/ (the published figures) with the same name in a rebuilt folder (default: plots/).

    python compare_to_reference.py [rebuilt_dir]

BYTE-IDENTICAL: same file. PIXEL-IDENTICAL: same image, different PNG metadata. DIFFERENT:
anything else. README.md, "Known differences from the published figures", lists every figure
known to differ and why.
"""
import hashlib, sys
from pathlib import Path
import numpy as np
from PIL import Image

ref_dir, new_dir = Path("Images/plots"), Path(sys.argv[1] if len(sys.argv) > 1 else "plots")
bad = 0
for ref in sorted(ref_dir.glob("*.png")):
    new = new_dir / ref.name
    if not new.exists():
        print(f"MISSING          {ref.name}"); bad += 1; continue
    if hashlib.md5(ref.read_bytes()).digest() == hashlib.md5(new.read_bytes()).digest():
        print(f"BYTE-IDENTICAL   {ref.name}"); continue
    a, b = (np.asarray(Image.open(p).convert("RGBA")).astype(int) for p in (ref, new))
    if a.shape == b.shape and not (a != b).any():
        print(f"PIXEL-IDENTICAL  {ref.name}"); continue
    print(f"DIFFERENT        {ref.name}"); bad += 1
sys.exit(1 if bad else 0)
