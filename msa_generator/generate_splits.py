import os
from glob import glob

splits_dir = os.path.abspath("./splits")
msa_dir = os.path.abspath("./data/msa")
unique = os.path.join(splits_dir, "unique")

if not os.path.exists(unique):
    lines = []
    for fasta in sorted(glob(os.path.join(msa_dir, "*", "?.fasta"))):
        pis, chain = fasta.split("/")[-2:]
        chain = chain[0]
        lines.append(pis + "/" + chain + "*\n")
    with open(unique, "w") as f:
        f.writelines(lines)

with open(unique, "r") as f:
    lines = f.readlines()

# for file in sorted(glob(os.path.join(msa_dir, "*", "*"))):
#     pis, chain = file.split("/")[-2:]
#     chain = chain[0]
#     if pis + "/" + chain not in avail:
#         print(file)
#         os.remove(file)

unique_ln = len(lines)
num_of_splits = 24
for i in range(0, unique_ln // num_of_splits + 1):
    with open(os.path.join(splits_dir, str(i)), "w") as f:
        for line in lines[i * num_of_splits : min(unique_ln, (i + 1) * num_of_splits)]:
            f.write(line)
