import sys, torch, torch.nn as nn
from torchvision import datasets, models, transforms as T
mode = sys.argv[1]
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
va = datasets.ImageFolder("dataset/val", tf)
m = models.resnet18(); m.fc = nn.Linear(512, len(va.classes))
m.load_state_dict(torch.load(f"models/resnet18_{mode}.pth")); m.eval()
cm = [[0] * len(va.classes) for _ in va.classes]
with torch.no_grad():
    for x, y in torch.utils.data.DataLoader(va, 32):
        for t, p in zip(y, m(x).argmax(1)): cm[t][p] += 1
print(va.classes, "| baris = asli, kolom = prediksi")
for r in cm: print(r)