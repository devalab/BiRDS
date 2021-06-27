import os

raw_dir = os.path.abspath("data/scPDB")


def make_clean_a3m():
    from string import ascii_lowercase

    remove_lowercase = str.maketrans(dict.fromkeys(ascii_lowercase))

    unique_file = os.path.abspath("splits/unique")
    with open(unique_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        pdb_id_struct, chain_id = line.strip().split()[0].split("/")
        chain_id = chain_id[:-1]
        pre = os.path.join(raw_dir, pdb_id_struct)
        if os.path.exists(os.path.join(pre, chain_id + ".a3m")):
            continue

        hhba3m = os.path.join(pre, chain_id + ".hhba3m")
        jaca3m = os.path.join(pre, chain_id + ".jaca3m")
        with open(os.path.join(pre, chain_id + ".aln"), "r") as f:
            aln_lines = f.readlines()

        sigh = {}
        if os.path.exists(hhba3m):
            with open(hhba3m, "r") as f:
                a3m_lines = f.readlines()
            for i in range(0, len(a3m_lines), 2):
                a3m_lines[i + 1] = a3m_lines[i + 1].translate(remove_lowercase)
                sigh[a3m_lines[i + 1]] = a3m_lines[i]

        if os.path.exists(jaca3m):
            with open(jaca3m, "r") as f:
                a3m_lines = f.readlines()
            for i in range(0, len(a3m_lines), 2):
                a3m_lines[i + 1] = a3m_lines[i + 1].translate(remove_lowercase)
                sigh[a3m_lines[i + 1]] = a3m_lines[i]

        new_a3m_lines = []
        for aln_line in aln_lines:
            new_a3m_lines.append(sigh[aln_line])
            new_a3m_lines.append(aln_line)
        with open(os.path.join(pre, chain_id + ".a3m"), "w") as f:
            f.writelines(new_a3m_lines)


def remove_extra_files(pre, chain_id, rm_fasta=False):
    fasta = os.path.join(pre, chain_id + ".fasta")
    aln = os.path.join(pre, chain_id + ".aln")
    jaca3m = os.path.join(pre, chain_id + ".jaca3m")
    hhba3m = os.path.join(pre, chain_id + ".hhba3m")
    a3m = os.path.join(pre, chain_id + ".a3m")
    if rm_fasta and os.path.exists(fasta):
        os.remove(fasta)
    if os.path.exists(aln):
        os.remove(aln)
    if os.path.exists(jaca3m):
        os.remove(jaca3m)
    if os.path.exists(hhba3m):
        os.remove(hhba3m)
    if os.path.exists(a3m):
        os.remove(a3m)


def remove_dna_rna_msas():
    import re

    def match(strg, search=re.compile(r"[^AGTCU]").search):
        return not bool(search(strg))

    for pdb_id_struct in sorted(os.listdir(raw_dir)):
        pre = os.path.join(raw_dir, pdb_id_struct)
        files = os.listdir(pre)
        for file in files:
            if file[2:] != "aln":
                continue
            chain = file[0]
            with open(os.path.join(pre, file), "r") as f:
                seq = f.readline().strip()
            if match(seq):
                remove_extra_files(pre, chain, True)


def remove_non_unique_files():
    unique_file = os.path.abspath("splits/unique")
    with open(unique_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        for extras in line[1:]:
            pdb_id_struct, chain = extras.split("/")
            chain = chain[:-1]
            pre = os.path.join(raw_dir, pdb_id_struct)
            remove_extra_files(pre, chain)


# remove_non_unique_files()
make_clean_a3m()
