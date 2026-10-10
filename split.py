import csv, shutil, os, collections
ROBOT_TRAIN = {"terang"}
by = collections.defaultdict(lambda: collections.defaultdict(list))
with open("dataset_raw/metadata.csv") as f:
    for r in csv.DictReader(f):
        by[r["kelas"]][r["kondisi_cahaya"]].append(r["nama_file"])
shutil.rmtree("dataset", ignore_errors=True)
for k, conds in by.items():
    n = {"train": 0, "val": 0}
    for c, files in conds.items():
        sp = "train" if (c == "kaggle" or c in ROBOT_TRAIN) else "val"
        os.makedirs(f"dataset/{sp}/{k}", exist_ok=True)
        for fn in files: shutil.copy(f"dataset_raw/{k}/{fn}", f"dataset/{sp}/{k}/{fn}")
        n[sp] += len(files)
    print(k, n)