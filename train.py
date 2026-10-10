import sys, time, json, os, torch, torch.nn as nn
from torchvision import datasets, models, transforms as T
from torch.utils.data import DataLoader

mode = sys.argv[1]                       # feature | partial | scratch
EPOCHS, BS = 10, 16
dev = "cuda" if torch.cuda.is_available() else "cpu"
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tr_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                   T.ColorJitter(0.3, 0.3, 0.3), T.ToTensor(), T.Normalize(mean, std)])
va_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean, std)])
tr = datasets.ImageFolder("dataset/train", tr_tf); va = datasets.ImageFolder("dataset/val", va_tf)
tl = DataLoader(tr, BS, shuffle=True); vl = DataLoader(va, BS)
NC = len(tr.classes)

m = models.resnet18(weights=None if mode == "scratch" else models.ResNet18_Weights.IMAGENET1K_V1)
m.fc = nn.Linear(m.fc.in_features, NC)
if mode == "feature":
    for p in m.parameters(): p.requires_grad = False
    for p in m.fc.parameters(): p.requires_grad = True
    groups = [{"params": m.fc.parameters(), "lr": 1e-3}]
elif mode == "partial":
    for p in m.parameters(): p.requires_grad = False
    for p in list(m.layer4.parameters()) + list(m.fc.parameters()): p.requires_grad = True
    groups = [{"params": m.layer4.parameters(), "lr": 1e-4}, {"params": m.fc.parameters(), "lr": 1e-3}]
else:
    groups = [{"params": m.parameters(), "lr": 1e-3}]
m.to(dev)
opt = torch.optim.Adam(groups)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.CrossEntropyLoss()

def set_train():
    m.train()
    for mod in m.modules():              # BatchNorm lapisan beku tetap eval
        if isinstance(mod, nn.BatchNorm2d) and not any(p.requires_grad for p in mod.parameters()):
            mod.eval()

hist, best, ep90, t0 = [], 0, None, time.time()
for ep in range(1, EPOCHS + 1):
    set_train(); tot = 0
    for x, y in tl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(); l = loss_fn(m(x), y); l.backward(); opt.step(); tot += l.item() * len(x)
    sched.step()
    m.eval(); ok = 0
    with torch.no_grad():
        for x, y in vl: ok += (m(x.to(dev)).argmax(1).cpu() == y).sum().item()
    acc = ok / len(va); tloss = tot / len(tr)
    hist.append({"epoch": ep, "train_loss": tloss, "val_acc": acc})
    if acc > best:
        best = acc; os.makedirs("models", exist_ok=True)
        torch.save(m.state_dict(), f"models/resnet18_{mode}.pth")
    if ep90 is None and acc >= 0.9: ep90 = ep
    print(f"[{mode}] ep{ep} loss {tloss:.3f} val_acc {acc:.3f}")

os.makedirs("results", exist_ok=True)
with open(f"results/{mode}.json", "w") as f:
    json.dump({"mode": mode, "best_val_acc": best, "train_time_s": round(time.time() - t0, 1),
               "epoch_acc90": ep90, "history": hist, "classes": tr.classes}, f, indent=2)