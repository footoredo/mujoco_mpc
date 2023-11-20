import os
import shutil
import sys
import subprocess


T = int(sys.argv[1])
name = sys.argv[2]
outcomes = []

for i in range(T):
    print(f"------ TEST {i} ------")
    try:
        os.remove("data.joblib")
    except FileNotFoundError:
        pass
    try:
        shutil.copyfile(f"{name}.data.joblib", "data.joblib")
    except FileNotFoundError:
        pass
    # subprocess.check_call(["python", "run_step.py"])
    subprocess.check_call(["python", f"run_step_{name}.py"])
    with open("outcome") as o:
        outcome = int(o.readline().strip())
        outcomes.append(outcome)

print(outcomes)
print(sum(outcomes))
