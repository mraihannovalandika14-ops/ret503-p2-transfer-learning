import cv2, sys, csv, os, datetime, numpy as np

CAM = 0                                   # ganti 1 atau 2 jika yang terbuka bukan kamera robot
kelas, kondisi = sys.argv[1], sys.argv[2]  # contoh: python capture.py Car terang
tgl = datetime.date.today().strftime("%Y%m%d")
out = f"dataset_raw/{kelas}"; os.makedirs(out, exist_ok=True)
meta = "dataset_raw/metadata.csv"
if not os.path.exists(meta):
    with open(meta, "w", newline="") as f:
        csv.writer(f).writerow(["nama_file", "kelas", "tanggal", "kondisi_cahaya"])

K = D = None
if os.path.exists("calib.npz"):
    c = np.load("calib.npz"); K, D = c["mtx"], c["dist"]   # sesuaikan nama key dengan calib.npz Anda

cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
cap.set(3, 640); cap.set(4, 480)          # harus sama dengan resolusi saat kalibrasi
n = len([f for f in os.listdir(out) if tgl in f and kondisi in f])

while True:
    ok, frame = cap.read()
    if not ok: break
    if K is not None: frame = cv2.undistort(frame, K, D)
    view = frame.copy()
    cv2.putText(view, f"{kelas}/{kondisi} saved:{n} [SPASI=simpan, q=keluar]", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("capture", view)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        n += 1
        nama = f"{kelas}_{tgl}_{kondisi}_{n:03d}.png"
        cv2.imwrite(f"{out}/{nama}", frame)
        with open(meta, "a", newline="") as f:
            csv.writer(f).writerow([nama, kelas, tgl, kondisi])
    elif k == ord("q"): break
cap.release(); cv2.destroyAllWindows()