import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

# ══════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════
input_image = r"D:\Bricklaying\B_ Video output\21\frame_00291.png"
output_dir  = r"D:\Bricklaying\B_ Video output\Depth map"
output_npy  = os.path.join(output_dir, "frame_00291_raw.npy")
output_vis  = os.path.join(output_dir, "frame_00291_depth.png")
os.makedirs(output_dir, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD IMAGE
# ══════════════════════════════════════════════════════════════════════════
img_bgr = cv2.imread(input_image)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W    = img_rgb.shape[:2]
print(f"Image size : {W} x {H} pixels")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — RUN MiDaS → GET DEPTH MAP
# ══════════════════════════════════════════════════════════════════════════
print("Loading MiDaS...")
midas     = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

with torch.no_grad():
    pred = midas(transform(img_rgb))
    pred = F.interpolate(pred.unsqueeze(1), size=(H, W),
                         mode="bicubic", align_corners=False).squeeze()

depth_map = pred.numpy()
np.save(output_npy, depth_map)

print(f"Depth map shape    : {depth_map.shape}")
print(f"MiDaS min (far)    : {depth_map.min():.2f}")
print(f"MiDaS max (close)  : {depth_map.max():.2f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — RUN YOLO → GET 17 KEYPOINTS (u, v)
# ══════════════════════════════════════════════════════════════════════════
from ultralytics import YOLO

model  = YOLO("best.pt")
result = model(img_bgr)[0]

kps_xy   = result.keypoints.xy[0].cpu().numpy()    # (17, 2)
kps_conf = result.keypoints.conf[0].cpu().numpy()  # (17,)

COCO17 = [
    "Nose", "Left_Eye", "Right_Eye", "Left_Ear", "Right_Ear",
    "Left_Shoulder", "Right_Shoulder", "Left_Elbow", "Right_Elbow",
    "Left_Wrist", "Right_Wrist", "Left_Hip", "Right_Hip",
    "Left_Knee", "Right_Knee", "Left_Ankle", "Right_Ankle"
]

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — READ Z FROM DEPTH MAP AT EACH KEYPOINT
# MiDaS gives Z directly — larger = closer, smaller = farther
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*55}")
print(f"{'#':<4} {'Keypoint':<20} {'u':>5} {'v':>5} {'conf':>6} {'Z (MiDaS)':>12}")
print(f"{'─'*55}")

kp_z = []   # store for visualisation

for i, name in enumerate(COCO17):
    u_c  = int(np.clip(kps_xy[i][0], 0, W-1))
    v_c  = int(np.clip(kps_xy[i][1], 0, H-1))
    conf = float(kps_conf[i])

    # This is what MiDaS gives you at this pixel
    Z = float(depth_map[v_c, u_c])

    kp_z.append((u_c, v_c, conf, Z, name))

    flag = "" if conf > 0.3 else "  ← low confidence"
    print(f"{i:<4} {name:<20} {u_c:>5} {v_c:>5} {conf:>6.2f} {Z:>12.2f}{flag}")

print(f"{'─'*55}")
print(f"\n  Larger Z = closer to camera")
print(f"  Smaller Z = farther from camera")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — VISUALISE: depth map + Z value at every keypoint
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 9))

# Left: original image with keypoint dots
axes[0].imshow(img_rgb)
axes[0].set_title("Original Image + YOLO Keypoints", fontsize=13)
for u_c, v_c, conf, Z, name in kp_z:
    if conf > 0.3:
        axes[0].plot(u_c, v_c, 'ro', markersize=7)
        axes[0].text(u_c+8, v_c, name.split("_")[-1],
                     color='yellow', fontsize=8, fontweight='bold')
axes[0].axis("off")

# Right: depth map with Z value printed at each keypoint
im = axes[1].imshow(depth_map, cmap="inferno")
axes[1].set_title("MiDaS Depth Map — Z value at each keypoint", fontsize=13)
plt.colorbar(im, ax=axes[1], label="MiDaS raw Z value")
for u_c, v_c, conf, Z, name in kp_z:
    if conf > 0.3:
        axes[1].plot(u_c, v_c, 'wo', markersize=7)
        axes[1].text(u_c+8, v_c, f"Z={Z:.0f}",
                     color='white', fontsize=8, fontweight='bold')
axes[1].axis("off")

plt.tight_layout()
plt.savefig(output_vis, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {output_vis}")