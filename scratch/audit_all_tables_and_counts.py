import json

with open("benchmark/task0_1_3_final_summary.json", "r", encoding="utf-8") as f:
    t_summary = json.load(f)

print("TASK 1 FEW-SHOT RESULTS FROM JSON:")
fs_data = t_summary.get("task1_few_shot_benchmark", {})
print(json.dumps(fs_data, indent=2))

with open("benchmark/final_fixes_results.json", "r", encoding="utf-8") as f:
    ff_data = json.load(f)

print("\nFINAL FIXES RESULTS FROM JSON:")
for k, v in ff_data.items():
    if "class_stats" in v:
        print(f"\n--- {k} ---")
        for cls, s in v["class_stats"].items():
            print(f"  {cls}: {s}")
