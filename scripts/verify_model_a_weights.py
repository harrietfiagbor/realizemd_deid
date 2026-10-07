"""
verify_model_a_weights.py

Provenance check for the Model A (Attention U-Net) weights file, so "same model"
claims are hash-verified instead of assumed from a matching filename.

Run this on whichever machine/pod is using Model A, and compare the printed MD5
against what the other side reports. Does NOT itself prove the file came from
the arkanivasarkar GitHub repo (that needs a fresh clone compared byte-for-byte,
not done here) -- it proves two parties are using the IDENTICAL file as each
other, which is the thing that actually matters for reconciling results.

Usage:
    python scripts/verify_model_a_weights.py "/path/to/weights.h5"
"""
import hashlib
import sys


def md5sum(path, chunk_size=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/verify_model_a_weights.py <path-to-weights-file>")
        sys.exit(1)
    path = sys.argv[1]
    print(f"File: {path}")
    print(f"MD5:  {md5sum(path)}")
    print()
    print("Local reference (Victoria, 2026-10-07): MD5 9cbd7640f82f083e9479240dcbb087cc")
    print("for arkan_unet/Trained models/retina_attentionUnet_150epochs.hdf5")
    print("(not independently verified against the arkanivasarkar GitHub repo itself --")
    print(" only confirms whether both sides are using the same file as each other.)")
