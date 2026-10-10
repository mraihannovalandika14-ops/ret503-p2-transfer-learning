import time, torch, numpy as np
from torchvision import models
x = torch.randn(1, 3, 224, 224)
for n, m in {"ResNet-18": models.resnet18(), "MobileNetV3-Small": models.mobilenet_v3_small()}.items():
    m.eval(); t = []
    with torch.inference_mode():
        for _ in range(20): m(x)
        for _ in range(100):
            s = time.perf_counter(); m(x); t.append((time.perf_counter() - s) * 1000)
    print(f"{n}: mean {np.mean(t):.1f} ms | p95 {np.percentile(t,95):.1f} ms | {1000/np.mean(t):.1f} FPS")