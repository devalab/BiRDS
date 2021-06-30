import os

import matplotlib.pyplot as plt
from collections import defaultdict


def pie_chart(labels, sizes, explode):
    # labels = ["Frogs", "Hogs", "Dogs", "Logs"]
    # sizes = [15, 30, 45, 10]
    # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots()
    ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct="%1.2f%%",
        shadow=True,
        startangle=90,
    )
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


data_dir = os.path.abspath("../data")

pic_to_pfam = defaultdict(set)
with open(os.path.join(data_dir, "pdbmap"), "r") as f:
    lines = [[el.strip() for el in line.strip().split(";")] for line in f.readlines()]
for line in lines:
    pic = line[0].lower() + "/" + line[1]
    # uni = line[4]
    pfam = line[3]
    pic_to_pfam[pic].add(pfam)
    # uni_to_pfam[uni].add(pfam)

pfam_to_pclan = defaultdict(set)
pclan_to_pfam = defaultdict(set)
with open(os.path.join(data_dir, "Pfam-A.clans.tsv"), "r") as f:
    lines = [line.strip().split() for line in f.readlines()]
for line in lines:
    pfam_to_pclan[line[0]].add(line[1])
    pclan_to_pfam[line[1]].add(line[0])

train = os.path.join(data_dir, "scPDB/uniprot_info.txt")
test = os.path.join(data_dir, "2018_scPDB/uniprot_info.txt")
with open(test, "r") as f:
    lines = [line.strip().split() for line in f.readlines()[1:]]
pic_to_uni = {line[0] + "/" + line[2]: line[3] for line in lines}

# pfam_to_uni = defaultdict(set)
pfam_to_pic = defaultdict(set)
pclan_to_pic = defaultdict(set)
for line in lines:
    pic = line[0] + "/" + line[2]
    # uni = line[3]
    pfam = pic_to_pfam[pic]
    # try:
    #     assert pfam == uni_to_pfam[uni]
    # except AssertionError:
    #     print(pic, uni, pfam, uni_to_pfam[uni])
    for el in pfam:
        pclan = pfam_to_pclan[el]
        for el2 in pclan:
            pclan_to_pic[el2].add(pic)
        pfam_to_pic[el].add(pic)
        # pfam_to_uni[el].add(uni)

pic_to_tax = defaultdict(set)
with open(os.path.join(data_dir, "pdb_chain_taxonomy.csv"), "r") as f:
    lines = [line.strip().split(",") for line in f.readlines()[2:]]
for line in lines:
    pic = line[0] + "/" + line[1]
    tax = line[2]
    pic_to_tax[pic].add(tax)

train = os.path.join(data_dir, "scPDB/uniprot_info.txt")
test = os.path.join(data_dir, "2018_scPDB/uniprot_info.txt")
with open(train, "r") as f:
    lines = [line.strip().split() for line in f.readlines()[1:]]
tax_to_pic = defaultdict(set)
for line in lines:
    pic = line[0] + "/" + line[2]
    if pic in pic_to_tax:
        tax = pic_to_tax[pic]
    else:
        print(pic)
        continue
    for el in tax:
        tax_to_pic[el].add(pic)


# scop_cla = os.path.join(data_dir, "scop-cla-latest.txt")
# with open(scop_cla, "r") as f:
#     lines = [line.strip().split() for line in f.readlines()[6:]]
# scop_cla = {line[3]: {el[:2]: el[3:] for el in line[-1].split(",")} for line in lines}

# scop_des = os.path.join(data_dir, "scop-des-latest.txt")
# with open(scop_des, "r") as f:
#     lines = [line.strip().split() for line in f.readlines()[6:]]
# scop_des = {line[0]: " ".join(line[1:]) for line in lines}

# labels = ["Globular", "Membrane", "Fibrous", "Others"]
# sizes = [0] * 4
# cnt = 0
# for pic, uni in pic_to_uni.items():
#     if uni in scop_cla:
#         sizes[int(scop_cla[uni]["TP"]) - 1] += 1
#     else:
#         cnt += 1
# # pie_chart(labels, sizes, (0, 0, 0, 0))
# print(sizes)
# print(cnt)

# import os
# import requests
# from xml.etree.ElementTree import fromstring

# pdb_mapping_url = "http://www.rcsb.org/pdb/rest/das/pdb_uniprot_mapping/alignment"
# uniprot_url = "http://www.uniprot.org/uniprot/{}.xml"


# def get_uniprot_accession_id(response_xml):
#     try:
#         root = fromstring(response_xml)
#     except:
#         return None
#     try:
#         return next(
#             el
#             for el in root.getchildren()[0].getchildren()
#             if el.attrib["dbSource"] == "UniProt"
#         ).attrib["dbAccessionId"]
#     except:
#         return None


# def map_pdb_to_uniprot(pdb_id):
#     pdb_mapping_response = requests.get(pdb_mapping_url, params={"query": pdb_id}).text
#     uniprot_id = get_uniprot_accession_id(pdb_mapping_response)
#     return {"pdb_id": pdb_id, "uniprot_id": uniprot_id}


# data_dir = os.path.abspath("../data")
# train = os.path.join(data_dir, "scPDB/info.txt")
# test = os.path.join(data_dir, "2018_scPDB/info.txt")
# mapping = os.path.join(data_dir, "pdbsws_chain.txt")

# with open(mapping, "r") as f:
#     lines = [line.strip().split() for line in f.readlines()]
# pic_to_uni = {line[0] + "/" + line[1]: line[2] for line in lines if len(line) == 3}

# with open(test, "r") as f:
#     lines = [line.strip().split() for line in f.readlines()[1:]]
# for i, line in enumerate(lines):
#     key = line[0] + "/" + line[2]
#     if key in pic_to_uni:
#         lines[i] = line[:3] + [pic_to_uni[key]] + line[3:]
#         # pass
#     else:
#         uni = map_pdb_to_uniprot(key.replace("/", "."))["uniprot_id"]
#         if uni:
#             lines[i] = line[:3] + [uni] + line[3:]
#         else:
#             print(line)
