import os
from collections import defaultdict

cfd = os.path.dirname(os.path.abspath(__file__))
msa_dir = os.path.join(cfd, "data/prep")
splits_dir = os.path.join(cfd, "splits")
unique = os.path.join(splits_dir, "unique")
file_types = ["fasta", "aln", "a3m", "pssm", "aap", "mtx", "ss2", "solv"]
cnts = defaultdict(lambda: [0] * len(file_types))


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def check_pssm(fname, seq):
    return file_len(fname) == 21


def check_mtx(fname, seq):
    return file_len(fname) == (len(seq) + 14)


def check_aap(fname, seq):
    return file_len(fname) == (len(seq) + 3)


def check_ss2(fname, seq):
    return file_len(fname) == (len(seq) + 2)


def check_solv(fname, seq):
    return file_len(fname) == len(seq)


for file in sorted(os.listdir(splits_dir)):
    with open(os.path.join(splits_dir, file), "r") as f:
        lines = f.readlines()
    for line in lines:
        pis, chain = line.strip().split()[0].split("/")
        chain = chain[0]
        pre = os.path.join(msa_dir, pis, chain + ".")
        with open(pre + "fasta", "r") as f:
            seq = "".join(f.read().splitlines()[1:]).strip()

        for i, ext in enumerate(file_types):
            ext_file = pre + ext
            if not (os.path.exists(ext_file) and os.stat(ext_file).st_size != 0):
                continue

            if "check_" + ext in globals():
                try:
                    tmp = globals()["check_" + ext](ext_file, seq)
                    if not tmp:
                        continue
                        # print(ext_file.split("/")[-2:])
                        # os.remove(ext_file)
                except Exception:
                    continue
                    # print(ext_file.split("/")[-2:])
                    # os.remove(ext_file)

            cnts[file][i] += 1

print("{:<10}".format("file"), end=" ")
for ext in file_types:
    print("{:<10}".format(ext), end=" ")

for key, lst in cnts.items():
    print("\n{:<10}".format(key), end=" ")
    for val in lst:
        print("{:<10}".format(val), end=" ")
print()
