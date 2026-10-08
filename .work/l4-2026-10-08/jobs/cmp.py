#!/usr/bin/env python3
"""Compare two identity arms (ident.sh output) by SHA-256, per directory."""
import hashlib, os, sys

def h(d):
    return {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
            for f in os.listdir(d)} if os.path.isdir(d) else {}

ref, arm = sys.argv[1], sys.argv[2]
for sub in ("cli", "str", "srv"):
    a, b = h(os.path.join(ref, sub)), h(os.path.join(arm, sub))
    c = sorted(set(a) & set(b))
    print("%s vs %s %s: files %d/%d common %d identical %d" %
          (os.path.basename(ref), os.path.basename(arm), sub, len(a), len(b), len(c),
           sum(a[k] == b[k] for k in c)))
