import os
from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Pool
from subprocess import PIPE, run

cfd = os.path.dirname(os.path.abspath(__file__))
hhlib = os.path.join(cfd, "hhsuite2")
os.environ["HHLIB"] = hhlib
aaprob = os.path.join(hhlib, "bin", "AlnAaProb")
a3m2mtx = os.path.join(hhlib, "scripts", "a3m2mtx.pl")
psipred = os.path.join(hhlib, "scripts", "hhb_psipred.pl")
wide = "-------------------------"


def run_cmd(pis, chain, cmds, file, name):
    log_out = ""
    try:
        if not os.path.exists(file) or os.stat(file).st_size == 0:
            log_out += " ".join([wide, pis, chain, name.upper(), wide, "\n"])
            start = datetime.now()
            if not isinstance(cmds[0], list):
                cmds = [cmds]
            for cmd in cmds:
                # print(cmd)
                if cmd[0] == aaprob:
                    output = run(cmd, stdout=open(file, "w"), stderr=PIPE)
                else:
                    output = run(cmd, stdout=PIPE, stderr=PIPE)
                    log_out += output.stdout.decode() + "\n"
                log_out += output.stderr.decode() + "\n"
                log_out += str(output.returncode) + "\n"
            log_out += " ".join(["Time Taken:", str(datetime.now() - start), "\n"])
            log_out += " ".join([wide, pis, chain, name.upper(), wide, "\n\n"])
    # print(log_out)
    except Exception:
        log_out += " ".join(["MYERR: ", pis, chain, name, " ".join(cmd), "\n"])
    return log_out


def task(file):
    log = ""
    start = datetime.now()
    pis, chain = file.strip().split("/")
    chain = chain[0]
    pre = os.path.join(raw_dir, pis)
    aln = os.path.join(pre, chain + ".aln")
    if not os.path.exists(aln):
        log += " ".join(["MYERR:", args.file, pis, chain, "alignment does not exist"])
        return log
    fasta = os.path.join(pre, chain + ".fasta")
    mtx = os.path.join(pre, chain + ".mtx")
    aap = os.path.join(pre, chain + ".aap")
    ss2 = os.path.join(pre, chain + ".ss2")
    solv = os.path.join(pre, chain + ".solv")

    cmd = [aaprob, aln]
    log += run_cmd(pis, chain, cmd, aap, "AAP")

    cmd = [a3m2mtx, aln, mtx, " -aln -neff 0"]
    log += run_cmd(pis, chain, cmd, mtx, "MTX")

    cmd = [psipred, fasta, " dummy ", ss2, mtx, solv]
    log += run_cmd(pis, chain, cmd, solv, "SS2 and SOLV")

    log += " ".join(["Total Time", pis, chain, str(datetime.now() - start), "\n\n\n\n"])
    return log


def extract_features_from_file(dataset_dir, file, ncpu):
    global raw_dir
    raw_dir = os.path.join(dataset_dir, "raw")

    with open(file, "r") as f:
        lines = f.readlines()

    with Pool(ncpu) as pool:
        for i, result in enumerate(pool.imap_unordered(task, lines)):
            print(result)
            print(i + 1, "PISC DONE")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Get secondary structure, amino-acid probabilites and solvent accessibility features",
        add_help=True,
    )
    parser.add_argument(
        "file",
        type=str,
        help="A file containing list of <pdb_id>/<chain_id>* lines for which to generate MSAs",
    )
    parser.add_argument(
        "--dataset-dir", default="../data/scPDB", type=str, help="Dataset directory",
    )
    parser.add_argument("-c", "--ncpu", default=1, type=int)
    args = parser.parse_args()
    print(args)
    extract_features_from_file(
        args.dataset_dir, os.path.join(args.dataset_dir, "splits", args.file), args.ncpu
    )
