import os
import numpy as np
from subprocess import run, PIPE

raw_dir = os.path.abspath("data/scPDB")


def delete_two_lines(original_file, line_number):
    """ Delete a line from a file at the given line number """
    is_skipped = False
    current_index = 0
    dummy_file = original_file + ".bak"
    # Open original file in read only mode and dummy file in write mode
    with open(original_file, "r") as read_obj, open(dummy_file, "w") as write_obj:
        # Line by line copy data from original file to dummy file
        for line in read_obj:
            # If current line number matches the given line number then skip copying
            if current_index != line_number and current_index != line_number + 1:
                write_obj.write(line)
            else:
                is_skipped = True
            current_index += 1

    # If any line is skipped then rename dummy file as original file
    if is_skipped:
        os.remove(original_file)
        os.rename(dummy_file, original_file)
    else:
        os.remove(dummy_file)


def fix_a3m(file, error):
    line_no = int(error.strip().split(" ")[-1]) - 2
    delete_two_lines(file, line_no)


def get_pssm():
    cmd1 = "./bin/esl-weight -p --amino --informat a2m -o weighted "
    cmd2 = "./bin/esl-alistat --weight --amino --icinfo icinfo --cinfo cinfo weighted"
    cmd3 = "rm weighted icinfo cinfo"
    for pdb_id_struct in sorted(os.listdir(raw_dir)):
        pre = os.path.join(raw_dir, pdb_id_struct)
        files = os.listdir(pre)
        for file in files:
            if file[2:] != "a3m":
                continue
            chain = file[0]
            if os.path.exists(os.path.join(pre, chain + ".pssm")):
                continue
            output = run(
                cmd1 + os.path.join(pre, file), shell=True, stderr=PIPE, stdin=PIPE
            )
            if output.returncode != 0:
                print(pdb_id_struct + " " + chain)
                fix_a3m(os.path.join(pre, file), output.stderr.decode())
                exit(1)
            flg = os.system(cmd2)
            if flg:
                print(pdb_id_struct + " " + chain)
                exit(1)
            i_icinfo = open("icinfo", "r")
            i_cinfo = open("cinfo", "r")
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
            np.savetxt(
                os.path.join(pre, chain + ".pssm"), np.stack(evos).T[:21], fmt="%1.5f"
            )
            os.system(cmd3)


get_pssm()
