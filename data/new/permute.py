import json, random

input_path = "test.jsonl"
output_path = "test_permuted.jsonl"

def permute_piis(piis_dict):
    """Return a new dict with shuffled key order."""
    if not isinstance(piis_dict, dict):
        return piis_dict
    items = list(piis_dict.items())
    random.shuffle(items)
    return dict(items)

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", encoding="utf-8") as fout:
    
    for line in fin:
        if not line.strip():
            continue
        entry = json.loads(line)
        if "piis" in entry:
            entry["piis"] = permute_piis(entry["piis"])
        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Saved permuted file to: {output_path}")