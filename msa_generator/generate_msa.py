import os
import sys
from shutil import rmtree
from string import ascii_lowercase
from subprocess import PIPE, run

import numpy as np

from .utils import fix_a3m

cfd = os.path.dirname(os.path.abspath(__file__))
tmpdir = os.path.join(cfd, "tmp")

sys.path.insert(0, os.path.join(cfd, "hhsuite2/scripts"))
from .hhsuite2.scripts.build_MSA import (
    build_MSA,
    check_db,
    make_tmpdir,
    parse_overwrite_option,
    read_one_sequence,
    refilter_aln,
    target_nf,
)

remove_lowercase = str.maketrans(dict.fromkeys(ascii_lowercase))
wide = "----------------------"

db_dict = dict(hmmsearchdb="",)
outdir = "."
overwrite = 0

cmd1 = os.path.join(cfd, "bin/esl-weight") + " -p --amino --informat a2m -o weighted "
icinfo = os.path.join(tmpdir, "icinfo")
cinfo = os.path.join(tmpdir, "cinfo")
cmd2 = (
    os.path.join(cfd, "bin/esl-alistat")
    + " --weight --amino --icinfo "
    + icinfo
    + " --cinfo "
    + cinfo
    + " weighted"
)
cmd3 = "rm weighted " + icinfo + " " + cinfo


def save_pssm(pis, chain, cnt=0):
    pre = os.path.join(raw_dir, pis)
    if os.path.exists(os.path.join(pre, chain + ".pssm")):
        return
    output = run(
        cmd1 + os.path.join(pre, chain + ".a3m"), shell=True, stderr=PIPE, stdin=PIPE
    )
    if output.returncode != 0:
        print(pre + " " + chain)
        if cnt <= 3:
            # Sometimes the a3m might not have proper sequences
            fix_a3m(os.path.join(pre, chain + ".a3m"), output.stderr.decode())
            save_pssm(pre, chain, cnt + 1)
        else:
            return
    flg = os.system(cmd2)
    if flg:
        print(pre + " " + chain + " Unable to get weighted info")
        return
    i_icinfo = open(icinfo, "r")
    i_cinfo = open(cinfo, "r")
    evos = []
    for buf_icinfo in range(9):
        buf_icinfo = i_icinfo.readline()
    for buf_cinfo in range(10):
        buf_cinfo = i_cinfo.readline()
    while buf_icinfo != "//\n":
        buf_icinfo_split = buf_icinfo.split()
        if buf_icinfo_split[0] != "-":
            ps = np.array([float(p) for p in buf_cinfo.split()[1:]])
            ps = ps / np.sum(ps)
            evo = np.append(ps, float(buf_icinfo_split[3]) / np.log2(20))
            evos.append(np.tile(evo, 2))
        buf_icinfo = i_icinfo.readline()
        buf_cinfo = i_cinfo.readline()
    i_icinfo.close()
    i_cinfo.close()
    np.savetxt(os.path.join(pre, chain + ".pssm"), np.stack(evos).T[:21], fmt="%1.5f")
    os.system(cmd3)


def make_a3m(pis, chain):
    pre = os.path.join(raw_dir, pis)
    if os.path.exists(os.path.join(pre, chain + ".a3m")):
        return

    hhba3m = os.path.join(pre, chain + ".hhba3m")
    jaca3m = os.path.join(pre, chain + ".jaca3m")
    with open(os.path.join(pre, chain + ".aln"), "r") as f:
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
    with open(os.path.join(pre, chain + ".a3m"), "w") as f:
        f.writelines(new_a3m_lines)


def generate_msa(pis, chain, ncpu=24):
    pre = os.path.join(raw_dir, pis)

    if os.path.exists(os.path.join(pre, chain + ".aln")):
        print(pis + "/" + chain + " alignment exists")
        return

    print(pis + "/" + chain + " processing now")
    chain_fasta = os.path.join(pre, chain + ".fasta")

    # check input format #
    query_fasta = os.path.abspath(chain_fasta)
    sequence = read_one_sequence(query_fasta)
    tmpdir_loc = make_tmpdir(os.path.join(tmpdir, pis + "_" + chain))
    prefix = os.path.splitext(query_fasta)[0]

    # start building MSA #
    nf = build_MSA(
        prefix,
        sequence,
        tmpdir_loc,
        db_dict,
        ncpu=ncpu,
        overwrite_dict=parse_overwrite_option(overwrite),
    )

    # filter final MSA if too large #
    # this will not improve contact accuracy. it is solely for making the
    # MSA not too large so that it is manageable for contact prediction
    if nf >= target_nf[-1]:
        nf = refilter_aln(prefix, tmpdir_loc)

    # clean up #
    if os.path.isdir(tmpdir_loc):
        rmtree(tmpdir_loc)


def generate_msas_from_file(dataset_dir, file, ncpu=24):
    data_dir = os.path.dirname(dataset_dir)
    db_dict["hhblitsdb"] = os.path.join(
        data_dir, "uniclust30_2017_10/uniclust30_2017_10"
    )
    db_dict["jackhmmerdb"] = os.path.join(data_dir, "uniref50.fasta")
    check_db(db_dict)

    global raw_dir
    raw_dir = os.path.join(dataset_dir, "raw")
    splits_dir = os.path.join(dataset_dir, "splits")

    # File should have 1 chain per line of the format
    # 1ABC/D*
    with open(os.path.join(splits_dir, file), "r") as f:
        lines = f.readlines()

    for file in lines:
        pis, chain = file.strip().split("/")
        chain = chain[:-1]

        # Create the MSAs using DeepMSA
        # Generates .aln, .jaca3m, .hhba3m, .jacaln, .hhbaln files
        print(wide + " GENERATING MSA " + wide)
        print(pis + "/" + chain)
        generate_msa(pis, chain, ncpu)
        print(pis + "/" + chain)
        print(wide + " GENERATED MSA " + wide)

        # An a3m file is needed instead of aln for generating pssm
        # So using .jaca3m and .hhba3m and the sequences present in .aln
        # the a3m file is made
        print(wide + " GENERATING A3M " + wide)
        print(pis + "/" + chain)
        make_a3m(pis, chain)
        print(pis + "/" + chain)
        print(wide + " GENERATED A3M " + wide)

        # Generate the PSSM using the a3m file created
        print(wide + " GENERATING PSSM " + wide)
        print(pis + "/" + chain)
        save_pssm(pis, chain)
        print(pis + "/" + chain)
        print(wide + " GENERATED PSSM " + wide)


if __name__ == "__main__":
    ncpu = 24
    if sys.argv[2].startswith("-ncpu="):
        ncpu = int(sys.argv[2][len("-ncpu=") :])
    data_dir = os.path.join(cfd, "../data")
    dataset_dir = os.path.join(data_dir, "scPDB")

    generate_msas_from_file(data_dir, dataset_dir, os.path.abspath(sys.argv[1]), ncpu)
