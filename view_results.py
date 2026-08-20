import pickle
from pprint import pprint

path = "outputs/geovision_results.pkl"

with open(path, "rb") as f:
    results = pickle.load(f)

print("=" * 70)
print("GEOVISION AI SAVED RESULTS")
print("=" * 70)

print("Type:", type(results))

if isinstance(results, dict):
    print("\nKeys:")
    for key in results:
        print(" -", key)

    print("\nFull results:")
    pprint(results, width=120)
else:
    print("\nResults:")
    pprint(results, width=120)