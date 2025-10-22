#!/usr/bin/env python3
# Validates dataset/test_out.csv against the challenge spec.

import sys, os
import pandas as pd
import numpy as np

def fail(msg):
    print(f"❌ {msg}"); sys.exit(1)
def ok(msg):
    print(f"✅ {msg}")

def main():
    test_csv = os.path.join("dataset","test.csv")
    sub_csv  = os.path.join("dataset","test_out.csv")

    if not os.path.exists(test_csv): fail(f"Missing {test_csv}")
    if not os.path.exists(sub_csv):  fail(f"Missing {sub_csv}")

    test = pd.read_csv(test_csv, dtype={"sample_id": str})
    sub  = pd.read_csv(sub_csv,  dtype={"sample_id": str})

    expected = ["sample_id","price"]
    if list(sub.columns) != expected:
        fail(f"Columns must be exactly {expected}, got {list(sub.columns)}")
    ok("Header/columns correct")

    if len(sub) != len(test):
        fail(f"Row count mismatch: test={len(test)} vs submission={len(sub)}")
    ok(f"Row count matches: {len(sub)}")

    if sub["sample_id"].duplicated().any():
        dups = sub.loc[sub["sample_id"].duplicated(),"sample_id"].unique()[:5]
        fail(f"Duplicate sample_id(s): {list(dups)}")
    ok("No duplicate sample_id values")

    if set(test["sample_id"]) != set(sub["sample_id"]):
        missing = list(set(test["sample_id"]) - set(sub["sample_id"]))[:5]
        extra   = list(set(sub["sample_id"]) - set(test["sample_id"]))[:5]
        fail(f"sample_id mismatch. Missing: {missing} | Extra: {extra}")
    ok("sample_id set exactly matches test.csv")

    # price must be numeric, finite, positive
    if not np.issubdtype(sub["price"].dtype, np.number):
        sub["price"] = pd.to_numeric(sub["price"], errors="coerce")
    if sub["price"].isna().any(): fail(f"'price' has {int(sub['price'].isna().sum())} NaNs")
    if not np.isfinite(sub["price"].values).all(): fail("'price' has non-finite values")
    if (sub["price"] <= 0).any(): fail(f"'price' has {(sub['price']<=0).sum()} non-positive values")
    ok("All 'price' values are numeric, finite, positive")

    if list(sub["sample_id"]) != list(test["sample_id"]):
        print("ℹ️ Note: submission order differs from test.csv (allowed).")

    ok("Submission looks good. Ready to upload ✅")

if __name__ == "__main__":
    main()
