import sys
import os
from shutil import rmtree

sys.path.insert(0, os.path.abspath("./hhsuite2/scripts"))
from hhsuite2.scripts.build_MSA import (
    build_MSA,
    check_db,
    make_tmpdir,
    parse_overwrite_option,
    read_one_sequence,
    refilter_aln,
    target_nf,
)

# command line argument parsing #
db_dict = dict(hmmsearchdb="",)
outdir = "."
overwrite = 0
ncpu = 24
if sys.argv[2].startswith("-ncpu="):
    ncpu = int(sys.argv[2][len("-ncpu=") :])

db_dict["hhblitsdb"] = os.path.abspath("./data/uniclust30_2017_10/uniclust30_2017_10")
db_dict["jackhmmerdb"] = os.path.abspath("./data/uniref50.fasta")
tmpdir = os.path.abspath("../tmp/" + sys.argv[1])
raw_dir = os.path.abspath("data/scPDB")
splits_dir = os.path.abspath("splits")
check_db(db_dict)

with open(os.path.join(splits_dir, sys.argv[1]), "r") as f:
    lines = f.readlines()

for file in lines:
    file, chain = file.strip().split("/")
    chain = chain[:-1]
    pre = os.path.join(raw_dir, file)

    if os.path.exists(os.path.join(pre, chain + ".aln")):
        print(file + "/" + chain + " alignment exists")
        continue

    print(file + "/" + chain + " processing now")
    chain_fasta = os.path.join(pre, chain + ".fasta")

    # check input format #
    query_fasta = os.path.abspath(chain_fasta)
    sequence = read_one_sequence(query_fasta)
    tmpdir = make_tmpdir(tmpdir)
    prefix = os.path.splitext(query_fasta)[0]

    # start building MSA #
    nf = build_MSA(
        prefix,
        sequence,
        tmpdir,
        db_dict,
        ncpu=ncpu,
        overwrite_dict=parse_overwrite_option(overwrite),
    )

    # filter final MSA if too large #
    # this will not improve contact accuracy. it is solely for making the
    # MSA not too large so that it is manageable for contact prediction
    if nf >= target_nf[-1]:
        nf = refilter_aln(prefix, tmpdir)

    # clean up #
    if os.path.isdir(tmpdir):
        rmtree(tmpdir)
