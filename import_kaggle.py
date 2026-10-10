import os, random, shutil, csv, glob
random.seed(42)
SRC, N = "Car-Bike-Dataset/train", 200
meta = "dataset_raw/metadata.csv"
os.makedirs("dataset_raw", exist_ok=True)
baru = not os.path.exists(meta)
with open(meta, "a", newline="") as fm:
    w = csv.writer(fm)
    if baru: w.writerow(["nama_file", "kelas", "tanggal", "kondisi_cahaya"])
    for kelas in os.listdir(SRC):
        files = [f for f in glob.glob(f"{SRC}/{kelas}/*.*") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(files)
        os.makedirs(f"dataset_raw/{kelas}", exist_ok=True)
        for i, f in enumerate(files[:N], 1):
            nama = f"{kelas}_kaggle_{i:03d}.jpg"
            shutil.copy(f, f"dataset_raw/{kelas}/{nama}")
            w.writerow([nama, kelas, "-", "kaggle"])
        print(kelas, min(N, len(files)))