#!/usr/bin/env python
import os
import sys

docstring = """
reindex_pdb.py startindex infile.pdb outfile.pdb
    Rearrange residue number of "infile.pdb" so the first residue start from
    "startindex". if the structure contains any missing residue, residue
    number gaps will be removed.

reindex_pdb.py seq.fasta  infile.pdb outfile.pdb
    Rearrange residue number of "infile.pdb" according to "seq.fasta" so the
    residue index is the same as that in "seq.fasta"

options:
    -clean={true,false} whether to remove lines other than "ATOM  " entries
"""


def reindex_pdb_by_index(startindex=1, PDBtxt=""):
    """
    reindex residue number of PDB format text

    options:
        startindex - index of first residue
        PDBtxt     - text of input PDB to be reindexed
    """
    PDBtxt_reindex = ""
    current_old_index = ""  # residue number in origin PDB
    warn_chainID = ""  # warning about new chain ID

    for line in PDBtxt.splitlines():
        if len(line) < 27 or (
            not line.startswith("ATOM  ")
            and not line.startswith("HETATM")
            and not line.startswith("TER")
        ):
            PDBtxt_reindex += line + "\n"
            continue
        elif not line[16] in ["A", " "]:  # alternative location identifier
            continue
        resSeq = line[22:27]  # residue sequence number
        current_chainID = line[21]  # chain identifier

        if not current_old_index:  # first residue encountered
            current_old_index = resSeq  # residue number in origin PDB
            current_new_index = int(startindex)
            chainID = current_chainID
            resSeq_new = str(current_new_index)
            resSeq_new = " " * (4 - len(resSeq_new)) + resSeq_new + " "
        elif current_chainID != chainID:
            if warn_chainID != current_chainID:
                sys.stderr.write("Warning! Discarding chain '%s'\n" % current_chainID)
                warn_chainID = current_chainID
            continue
        elif resSeq != current_old_index:
            current_new_index += 1
            current_old_index = resSeq
            resSeq_new = str(current_new_index)
            resSeq_new = " " * (4 - len(resSeq_new)) + resSeq_new + " "
        PDBtxt_reindex += line[:16] + " " + line[17:22] + resSeq_new + line[27:] + "\n"
    return PDBtxt_reindex


def find_next_amino_acid(sequence_ref, sequence_pdb, startindex=0):
    """find the first instance of amino acid whose index is greater than
    startindex. index is one-based. return -1 if not found. return negative
    if the position is gap in sequence_ref
    """
    startindex = abs(startindex)
    for resi in range(startindex, len(sequence_pdb)):
        if sequence_pdb[resi] != "-":
            return (resi + 1) * (-1) ** (sequence_ref[resi] == "-")
    return 0


def reindex_pdb_by_sequence(sequence_ref, PDBtxt, chain_id):
    """NWalign "sequence_pdb" to "sequence_ref" and renumber residue index
    according to sequence_ref.

    unaligned residues in sequence_pdb will be removed.
    only ATOM or MSE will be parsed
    """
    # convert PDB to fasta ####
    from pbsp.utils.pdb2fasta import pdbtxt2seq

    header_list, sequence_list = pdbtxt2seq(PDBtxt, PERMISSIVE="MSE", allowX=True)
    index = None
    for i, el in enumerate(header_list):
        if el[-1] == chain_id:
            index = i
            break
    if index is None:
        return None
    sequence_pdb = sequence_list[index]

    # perform NeedlemanWunsch ####
    from pbsp.utils.NWalign import calcualte_score_gotoh, trace_back_gotoh

    # idir (DP path); jpV (Horizontal jump number); jpH (Vertical jump number)
    idir, jpV, jpH = calcualte_score_gotoh(sequence_ref, sequence_pdb)
    # sequenceA (aligned f1); sequenceB (aligned f2)
    sequence_ref, sequence_pdb = trace_back_gotoh(
        idir, jpV, jpH, sequence_ref, sequence_pdb
    )

    # reindex PDB text ####
    PDBtxt_reindex = ""
    current_old_index = ""  # residue number in origin PDB

    for line in PDBtxt.splitlines():
        if (
            len(line) < 27
            or line[16] not in ["A", " "]
            or (
                not line.startswith("ATOM  ")
                and not line.startswith("TER")
                and not (line.startswith("HETATM") and line[17:20] == "MSE")
            )
            or line[21] != chain_id
        ):
            # PDBtxt_reindex += line + "\n"
            continue
        resSeq = line[22:27]  # residue sequence number

        if not current_old_index:  # first residue encountered
            current_old_index = resSeq  # residue number in origin PDB
            current_new_index = find_next_amino_acid(sequence_ref, sequence_pdb)
            resSeq_new = str(current_new_index)
            resSeq_new = " " * (4 - len(resSeq_new)) + resSeq_new + " "
        elif resSeq != current_old_index:
            current_new_index = find_next_amino_acid(
                sequence_ref, sequence_pdb, current_new_index
            )
            current_old_index = resSeq
            resSeq_new = str(current_new_index)
            resSeq_new = " " * (4 - len(resSeq_new)) + resSeq_new + " "

        if current_new_index == 0:  # reach the end of alignment
            break
        elif current_new_index > 0:  # no gap in sequence_ref:
            PDBtxt_reindex += (
                line[:16] + " " + line[17:22] + resSeq_new + line[27:] + "\n"
            )
    return PDBtxt_reindex


def reindex_pdb(startindex, infile, clean=True):
    """parse PDB file "infile", reindex it according to start index or
    sequence file "startindex", and return the text of renumbered PDB
    """
    from pbsp.utils.NWalign import code_with_modified_residues

    fp = open(infile, "rU")
    PDBtxt = ""
    for line in fp.read().splitlines():
        if line.startswith("END"):
            if clean:
                line = line.replace("ENDMDL", "END   ")
            PDBtxt += line + "\n"
            break
        if (
            (
                line.startswith("ATOM  ")
                and line[17:20] in code_with_modified_residues.keys()
            )
            or (
                line.startswith("TER")
                and line[17:20] in code_with_modified_residues.keys()
            )
            or (
                clean is False
                and not line[:6]
                in [
                    "DBREF ",
                    "SEQADV",
                    "MODRES",
                    "HELIX ",
                    "SHEET ",
                    "SSBOND",
                    "SITE  ",
                ]
            )
        ):
            PDBtxt += line + "\n"
    fp.close()

    if os.path.isfile(startindex):
        from pbsp.utils.NWalign import readFastaOrRawSequence

        sequence_ref = readFastaOrRawSequence(startindex).replace("-", "")
        # Replace celenocystein with X (U with X)
        sequence_ref = sequence_ref.replace("U", "X")
        chain_id = startindex[-7:-6]
        PDBtxt_reindex = reindex_pdb_by_sequence(sequence_ref, PDBtxt, chain_id)
    else:
        PDBtxt_reindex = reindex_pdb_by_index(startindex, PDBtxt)
    return PDBtxt_reindex


if __name__ == "__main__":
    # parse commandline arguments ####
    clean = True

    argv = []
    for arg in sys.argv[1:]:
        if arg.startswith("-clean="):
            clean = arg[len("-clean=") :].lower() == "true"
        elif arg.startswith("-"):
            sys.stderr.write("ERROR! Unknown option %s\n" % arg)
            exit()
        else:
            argv.append(arg)

    if len(argv) < 2:
        sys.stderr.write(docstring)
        exit()

    # parse PDB file ####
    PDBtxt_reindex = reindex_pdb(argv[0], argv[1], clean)
    if PDBtxt_reindex is None:
        print("The given sequence is a DNA/RNA sequence")
        exit(1)

    # write PDB file ####
    if len(argv) > 1:
        fp = open(argv[2], "w")
        fp.write(PDBtxt_reindex)
        fp.close()
