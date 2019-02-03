import re
import os
from collections import defaultdict

data = defaultdict(lambda: defaultdict(dict))

files = os.listdir("data")
regex1 = re.compile("trained_model-actor_target_([0-9]*)-agent-A-([0-9]*).pth")
paramsob = {'aaaa': 1, 'bbbb': 2}

for filename in files:
    m = regex1.search(filename)
    if m:
        for k, v in paramsob.items():
            data[m.group(1)][k][int(m.group(2))] = v
    else:
        print(f"     {filename}")
