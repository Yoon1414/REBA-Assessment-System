# ==========================================================
# app.py — Streamlit Main Application
# Imports: reba.py, angle.py, visualization.py, export.py
# Supports: Single image, Video, Folder of images
# Panels: panel1_skeleton, panel2_depth, panel3_coordinates, panel4_reba
# ==========================================================

import io
import os
import cv2
import glob
import datetime
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

from ultralytics import YOLO
import torch

# ── Import project modules ────────────────────────────────
from reba          import compute_reba, risk_color_bgr
from angle         import (compute_angles_2d, compute_angles_3d,
                            KP_CONF_THRESHOLD, COCO17_NAMES)
from visualization import draw_person
from export        import build_excel, build_pdf
from panels.panel1_skeleton    import render_panel1
from panels.panel2_depth       import render_panel2
from panels.panel3_coordinates import render_panel3
from panels.panel4_reba        import render_panel4

# ============================================================
# HELPER — Render all 4 panels for one person
# MUST be defined before it is called below
# ============================================================

def _render_all_panels(frame, depth_vis, depth, kps, kpc,
                        tid, conf, result, use_midas, w, h):
    """
    Render Panel 1-4 for a single detected person.

    Parameters
    ----------
    frame      : annotated BGR image (already drawn on)
    depth_vis  : colored depth map BGR image (or None if 2D mode)
    depth      : raw depth numpy array (or None)
    kps        : (17,2) keypoint pixel coordinates
    kpc        : (17,) keypoint confidences
    tid        : track / worker ID
    conf       : YOLO detection confidence
    result     : dict from compute_reba()
    use_midas  : bool
    w, h       : frame width / height
    """

    tab1, tab2, tab3, tab4 = st.tabs([
        "🦴 Skeleton",
        "🌈 Depth Map",
        "📍 Coordinates",
        "📊 REBA Workflow",
    ])

    # ── Panel 1: Skeleton Detection ───────────────────────
    with tab1:
        render_panel1(
            annotated_img = frame,
            person_id     = tid,
            confidence    = conf,
            reba_score    = result["REBA_final"],
            risk_level    = result["risk"],
        )

    # ── Panel 2: Depth Map ────────────────────────────────
    with tab2:
        if use_midas and depth_vis is not None and depth is not None:
            z_values = []
            for (px, py) in kps:
                ui = int(np.clip(px, 0, w - 1))
                vi = int(np.clip(py, 0, h - 1))
                z_values.append(float(depth[vi, ui]))
            render_panel2(
                depth_image = depth_vis,
                z_values    = z_values,
            )
        else:
            st.info(
                "🔵 MiDaS 3D Depth is **disabled**. "
                "Enable it in System Settings to see the depth map."
            )

    # ── Panel 3: Joint Coordinates ────────────────────────
    with tab3:
        if use_midas and depth is not None:
            coord_rows = []
            for ji, (px, py) in enumerate(kps):
                ui = int(np.clip(px, 0, w - 1))
                vi = int(np.clip(py, 0, h - 1))
                Z  = float(depth[vi, ui])
                coord_rows.append({
                    "Joint":       COCO17_NAMES[ji],
                    "u (pixel x)": round(float(px), 1),
                    "v (pixel y)": round(float(py), 1),
                    "z (depth)":   round(Z, 4),
                    "Confidence":  round(float(kpc[ji]), 3),
                })
        else:
            coord_rows = []
            for ji, (px, py) in enumerate(kps):
                coord_rows.append({
                    "Joint":       COCO17_NAMES[ji],
                    "u (pixel x)": round(float(px), 1),
                    "v (pixel y)": round(float(py), 1),
                    "Confidence":  round(float(kpc[ji]), 3),
                })
        coord_df = pd.DataFrame(coord_rows)
        render_panel3(coord_df=coord_df)

    # ── Panel 4: REBA Assessment Workflow ─────────────────
    with tab4:
        render_panel4(
            neck       = result["neck_bin"],
            trunk      = result["trunk_bin"],
            leg        = result["leg_bin"],
            upper_arm  = result["upper_bin"],
            lower_arm  = result["lower_bin"],
            wrist      = result["wrist_bin"],
            table_a    = result["A_posture"],
            load_force = result["force_score"],
            score_a    = result["Score_A"],
            table_b    = result["B_posture"],
            coupling   = result["coupling"],
            score_b    = result["Score_B"],
            table_c    = result["C_score"],
            activity   = result["activity"],
            reba_score = result["REBA_final"],
            risk_level = result["risk"],
        )


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(layout="wide", page_title="REBA Assessment System")
st.title("🏗️ REBA Assessment System")
st.markdown("YOLOv11 Fine-tuned | Confidence-aware | 2D / 3D MiDaS | Batch + Single | PDF + Excel")
st.divider()

# ============================================================
# SIDEBAR — Settings
# ============================================================

st.sidebar.header("⚙️ System Settings")

use_midas = st.sidebar.toggle(
    "🔵 Use MiDaS 3D Depth", value=False,
    help="OFF = 2D pixel angles (stable, recommended)\nON = 3D with MiDaS depth")

if use_midas:
    depth_variant = st.sidebar.selectbox(
        "MiDaS model",
        ["MiDaS_small", "DPT_Hybrid"],
        help="MiDaS_small = faster | DPT_Hybrid = more accurate"
    )

pose_weights = st.sidebar.text_input(
    "YOLO model path",
    value="best.pt")   # relative path — place best.pt in project root

kp_conf_thresh = st.sidebar.slider(
    "Keypoint confidence threshold",
    min_value=0.1, max_value=0.9, value=0.3, step=0.05,
    help="Higher = stricter. Low-confidence joints use neutral defaults.")

st.sidebar.divider()

# ============================================================
# SIDEBAR — Manual REBA Adjustments
# ============================================================

st.sidebar.header("📋 Manual REBA Adjustments")

st.sidebar.subheader("Step 3 — Legs")
one_leg_raised = st.sidebar.checkbox("One leg raised / walking (+1)")

st.sidebar.subheader("Step 5 — Force / Load")
force_load = st.sidebar.selectbox("Load:", options=[0,1,2],
    format_func=lambda x: {
        0:"< 5 kg (+0)", 1:"5–10 kg (+1)", 2:"> 10 kg (+2)"}[x])
shock = st.sidebar.checkbox("Shock / rapid force (+1)")
st.sidebar.info(f"Force score: **+{force_load + int(shock)}**")

st.sidebar.subheader("Step 7a — Upper Arm")
arm_supported = st.sidebar.checkbox("Arm supported / leaning (-1)")

st.sidebar.subheader("Step 9a — Wrist")
wrist_twisted = st.sidebar.checkbox("Wrist twisted (+1)")

st.sidebar.subheader("Step 11 — Coupling / Grip")
coupling = st.sidebar.selectbox("Grip quality:", options=[0,1,2,3],
    format_func=lambda x: {
        0:"Good — well fitting (+0)",
        1:"Fair — acceptable (+1)",
        2:"Poor — not ideal (+2)",
        3:"Unacceptable — no handle (+3)"}[x])
st.sidebar.info(f"Coupling: **+{coupling}**")

st.sidebar.subheader("Step 13 — Activity")
act_static   = st.sidebar.checkbox("Static posture > 1 min (+1)")
act_repeated = st.sidebar.checkbox("Repeated actions > 4x/min (+1)")
act_rapid    = st.sidebar.checkbox("Rapid large changes (+1)")
act_total    = int(act_static) + int(act_repeated) + int(act_rapid)
st.sidebar.info(f"Activity: **+{act_total}**")

st.sidebar.divider()

# ============================================================
# SIDEBAR — Input Mode
# ============================================================

st.sidebar.header("📁 Input")

input_mode = st.sidebar.radio(
    "Input mode",
    ["📂 Folder Path (batch)", "🖼️ Single Image", "🎥 Video"],
    help="Folder = process all images in folder at once"
)

# ── Option 1: Folder Path ─────────────────────────────────
frames      = []
frame_names = []

if input_mode == "📂 Folder Path (batch)":
    st.sidebar.markdown("**Paste your image folder path:**")
    folder_path = st.sidebar.text_input(
        "Folder path",
        value="",   # paste your image folder path here
        help="All JPG/PNG images in this folder will be processed"
    )

    if folder_path and os.path.isdir(folder_path):
        all_files = os.listdir(folder_path)
        img_paths = []
        for f in all_files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_paths.append(os.path.join(folder_path, f))
        img_paths = sorted(img_paths)

        if img_paths:
            st.sidebar.success(f"✅ Found **{len(img_paths)} images**")
            st.sidebar.caption(f"First: {os.path.basename(img_paths[0])}")
            st.sidebar.caption(f"Last:  {os.path.basename(img_paths[-1])}")
        else:
            st.sidebar.warning("⚠️ No images found in this folder")
            img_paths = []

    elif folder_path:
        st.sidebar.error("❌ Folder not found. Check the path.")
        img_paths = []
    else:
        img_paths = []

    file = None

# ── Option 2: Single Image ────────────────────────────────
elif input_mode == "🖼️ Single Image":
    file = st.sidebar.file_uploader(
        "Upload image",
        type=["jpg","jpeg","png"])
    img_paths = []

# ── Option 3: Video ───────────────────────────────────────
else:
    file = st.sidebar.file_uploader(
        "Upload video",
        type=["mp4","mov","avi"])
    img_paths = []

# ============================================================
# RUN BUTTON
# ============================================================

ready = False
if input_mode == "📂 Folder Path (batch)" and img_paths:
    ready = True
    st.sidebar.markdown(f"**Ready to process {len(img_paths)} images**")
elif input_mode in ["🖼️ Single Image", "🎥 Video"] and file:
    ready = True

run_btn = st.sidebar.button("▶️ Run REBA Analysis",
                             disabled=not ready,
                             type="primary")

# ============================================================
# MAIN — Processing
# ============================================================

if run_btn and ready:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Device: **{device.upper()}** | "
            f"Mode: **{'3D MiDaS' if use_midas else '2D Pixel'}** | "
            f"Input: **{input_mode}**")

    # Load YOLO
    with st.spinner("Loading YOLO model..."):
        model = YOLO(pose_weights)

    # Load MiDaS (if enabled)
    midas = None; transform = None
    if use_midas:
        with st.spinner("Loading MiDaS depth model..."):
            midas     = torch.hub.load("intel-isl/MiDaS", depth_variant)
            midas.to(device).eval()
            transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    # ── Load frames ───────────────────────────────────────
    frames      = []
    frame_names = []

    if input_mode == "📂 Folder Path (batch)":
        st.info(f"Loading {len(img_paths)} images from folder...")
        for path in img_paths:
            img = cv2.imread(path)
            if img is not None:
                frames.append(img)
                frame_names.append(os.path.basename(path))
        st.success(f"✅ Loaded **{len(frames)}** images successfully")

    elif input_mode == "🖼️ Single Image":
        img = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        frames.append(img)
        frame_names.append(file.name)

    else:  # Video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        fcount = 0
        while True:
            ret, frm = cap.read()
            if not ret: break
            frames.append(frm)
            frame_names.append(f"frame_{fcount:04d}")
            fcount += 1
        cap.release()

    # ── Processing loop ───────────────────────────────────
    scores_all       = []
    coords_all       = []
    annotated_frames = []

    prog  = st.progress(0, text="Starting...")
    total = len(frames)

    if input_mode == "📂 Folder Path (batch)":
        st.subheader(f"📸 Processing {total} images")
        img_cols = st.columns(3)
        col_idx  = 0

    for fid, frame in enumerate(frames):
        fname = frame_names[fid]
        prog.progress(
            (fid + 1) / total,
            text=f"Processing {fid+1}/{total}: {fname}")

        h, w   = frame.shape[:2]
        cx, cy = w / 2, h / 2
        fx = fy = float(w)

        # YOLO inference
        if input_mode == "🎥 Video":
            results = model.track(frame, persist=True, verbose=False)
        else:
            results = model.predict(frame, verbose=False)

        if results[0].keypoints is None:
            annotated_frames.append((fid, frame.copy()))
            continue

        kps_all    = results[0].keypoints.xy.cpu().numpy()
        kpconf_all = results[0].keypoints.conf.cpu().numpy()
        boxes      = results[0].boxes.xyxy.cpu().numpy()
        confs      = results[0].boxes.conf.cpu().numpy()

        if (input_mode == "🎥 Video" and results[0].boxes.id is not None):
            ids = results[0].boxes.id.cpu().numpy()
        else:
            ids = np.arange(1, len(kps_all) + 1)

        # MiDaS depth map
        depth     = None
        depth_vis = None
        if use_midas and midas is not None:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp     = transform(img_rgb).to(device)
            with torch.no_grad():
                d     = midas(inp)
                depth = torch.nn.functional.interpolate(
                    d.unsqueeze(1), size=(h, w),
                    mode="bicubic", align_corners=False
                ).squeeze().cpu().numpy()
            depth_norm = cv2.normalize(depth, None, 0, 255,
                                       cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis  = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

        # ── Per-person processing ──────────────────────────
        for pidx, kps in enumerate(kps_all):
            tid  = int(ids[pidx])
            conf = float(confs[pidx])
            kpc  = kpconf_all[pidx]
            if conf < 0.3:
                continue

            # Angles
            if use_midas and depth is not None:
                angles = compute_angles_3d(
                    kps, kpc, depth, cx, cy, fx, fy, w, h)
            else:
                angles = compute_angles_2d(kps, kpc)

            # REBA score
            result = compute_reba(
                angles,
                one_leg_raised = one_leg_raised,
                force_load     = force_load,
                shock          = shock,
                arm_supported  = arm_supported,
                coupling       = coupling,
                wrist_twisted  = wrist_twisted,
                act_static     = act_static,
                act_repeated   = act_repeated,
                act_rapid      = act_rapid,
            )

            # Draw on frame
            frame = draw_person(
                frame, kps, kpc, boxes[pidx],
                tid, conf, result,
                kp_conf_threshold=kp_conf_thresh)

            # Store REBA row
            scores_all.append([
                fid, fname, tid, round(conf, 2),
                result["trunk_fwd"],       result["neck_fwd"],
                result["knee_bend"],       result["upper_arm"],
                result["elbow"],           result["wrist_dev"],
                result["trunk_twisted"],   result["neck_twisted"],
                result["shoulder_raised"], result["arm_abducted"],
                result["trunk_bin_raw"],   result["trunk_adj"],   result["trunk_bin"],
                result["neck_bin_raw"],    result["neck_adj"],    result["neck_bin"],
                result["leg_bin"],
                result["upper_bin_raw"],   result["upper_adj"],   result["upper_bin"],
                result["lower_bin"],       result["wrist_bin"],
                result["A_posture"],       result["force_score"], result["Score_A"],
                result["B_posture"],       result["coupling"],    result["Score_B"],
                result["C_score"],         result["activity"],    result["REBA_final"],
                result["risk"],
            ])

            # Store coordinates
            row_2d = [fid, fname, tid]
            for ji, (px, py) in enumerate(kps):
                row_2d += [round(float(px), 2), round(float(py), 2),
                            round(float(kpc[ji]), 3)]
            coord_entry = {"2d": row_2d}

            if use_midas and depth is not None:
                row_3d = [fid, fname, tid]
                for (px, py) in kps:
                    ui = int(np.clip(px, 0, w - 1))
                    vi = int(np.clip(py, 0, h - 1))
                    Z  = float(depth[vi, ui])
                    row_3d += [round((px - cx) * Z / fx, 4),
                                round((py - cy) * Z / fy, 4),
                                round(Z, 4)]
                coord_entry["3d"] = row_3d

            coords_all.append(coord_entry)

            # ── Render all 4 panels ────────────────────────
            if input_mode == "📂 Folder Path (batch)":
                with st.expander(
                    f"📋 Frame {fid+1} | {fname} | Worker ID {tid} "
                    f"| REBA {result['REBA_final']} — {result['risk']}",
                    expanded=False
                ):
                    _render_all_panels(
                        frame, depth_vis, depth, kps, kpc,
                        tid, conf, result, use_midas, w, h
                    )
            else:
                st.markdown(
                    f"#### 🧍 Worker ID {tid} | Frame {fid+1} — {fname}"
                )
                _render_all_panels(
                    frame, depth_vis, depth, kps, kpc,
                    tid, conf, result, use_midas, w, h
                )

        annotated_frames.append((fid, frame.copy()))

        # ── Display frame thumbnail ────────────────────────
        if input_mode == "📂 Folder Path (batch)":
            disp_w = 600
            disp_h = int(frame.shape[0] * disp_w / frame.shape[1])
            display = cv2.resize(frame, (disp_w, disp_h))
            with img_cols[col_idx % 3]:
                st.image(display, channels="BGR",
                         caption=fname, use_container_width=True)
            col_idx += 1

        elif input_mode in ["🖼️ Single Image", "🎥 Video"]:
            disp_w = 800
            disp_h = int(frame.shape[0] * disp_w / frame.shape[1])
            display = cv2.resize(frame, (disp_w, disp_h))
            col_l, col_c, col_r = st.columns([1, 4, 1])
            with col_c:
                st.image(display, channels="BGR",
                         caption=f"{fname} — Frame {fid+1}",
                         use_container_width=True)

    prog.empty()

    # ============================================================
    # RESULTS DASHBOARD
    # ============================================================

    if not scores_all:
        st.warning("⚠️ No people detected. Check YOLO path and input.")
    else:
        cols = [
            "Frame_ID", "Filename", "TrackID", "Person_conf",
            "Trunk_fwd°", "Neck_fwd°", "Knee_bend°",
            "UpperArm°", "Elbow°", "Wrist_dev°",
            "Trunk_twisted", "Neck_twisted",
            "Shoulder_raised", "Arm_abducted",
            "Trunk_raw", "Trunk_adj", "Trunk_FINAL",
            "Neck_raw", "Neck_adj", "Neck_FINAL",
            "Leg_bin",
            "Upper_raw", "Upper_adj", "Upper_FINAL",
            "Lower_bin", "Wrist_bin",
            "A_posture", "Force", "Score_A",
            "B_posture", "Coupling", "Score_B",
            "Table_C", "Activity", "REBA_Final", "Risk"
        ]
        df = pd.DataFrame(scores_all, columns=cols)

        st.divider()
        st.subheader("📊 REBA Results Dashboard")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Images/Frames", total)
        c2.metric("People Detected", df["TrackID"].nunique())
        c3.metric("Total Detections", len(df))
        c4.metric("Avg REBA", round(df["REBA_Final"].mean(), 1))
        c5.metric("Max REBA", df["REBA_Final"].max())

        st.subheader("Risk Level Summary")
        rc = df["Risk"].value_counts().reset_index()
        rc.columns = ["Risk Level", "Count"]
        risk_order = ["Negligible", "Low", "Medium", "High", "Very High"]
        rc["Risk Level"] = pd.Categorical(
            rc["Risk Level"], categories=risk_order, ordered=True)
        rc = rc.sort_values("Risk Level")
        st.dataframe(rc, use_container_width=True)

        st.subheader("REBA Score Distribution")
        st.bar_chart(df["REBA_Final"].value_counts().sort_index())

        st.subheader("Angle + Bin Breakdown (All Detections)")
        st.dataframe(df, use_container_width=True)

        # ── Downloads ──────────────────────────────────────
        st.divider()
        st.subheader("⬇️ Download Results")

        with st.spinner("Building Excel..."):
            excel_buf = build_excel(df, coords_all)

        with st.spinner("Building PDF report (all images)..."):
            pdf_buf = build_pdf(df, annotated_frames)

        csv_buf = io.BytesIO()
        csv_buf.write(df.to_csv(index=False).encode())
        csv_buf.seek(0)

        import zipfile
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"REBA_results_{timestamp}.xlsx", excel_buf.getvalue())
            zf.writestr(f"REBA_report_{timestamp}.pdf",   pdf_buf.getvalue())
            zf.writestr(f"REBA_scores_{timestamp}.csv",   csv_buf.getvalue())
        zip_buf.seek(0)

        st.download_button(
            label="📦 Download All Results (ZIP)",
            data=zip_buf,
            file_name=f"REBA_results_{timestamp}.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )
        st.caption(
            f"ZIP contains:\n"
            f"📊 REBA_results_{timestamp}.xlsx  "
            f"(Sheet 1: REBA Scores | Sheet 2: YOLO 2D coords | Sheet 3: MiDaS 3D coords)\n"
            f"📄 REBA_report_{timestamp}.pdf  "
            f"(All {len(annotated_frames)} annotated images + score tables)\n"
            f"📋 REBA_scores_{timestamp}.csv  "
            f"(Raw scores for further analysis)"
        )

        st.markdown("**Or download individually:**")
        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            excel_buf.seek(0)
            st.download_button(
                label="📊 Excel only",
                data=excel_buf,
                file_name=f"REBA_results_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.caption("Coordinates + REBA scores")

        with dl2:
            pdf_buf.seek(0)
            st.download_button(
                label="📄 PDF only",
                data=pdf_buf,
                file_name=f"REBA_report_{timestamp}.pdf",
                mime="application/pdf"
            )
            st.caption(f"All {len(annotated_frames)} annotated images")

        with dl3:
            csv_buf.seek(0)
            st.download_button(
                label="📋 CSV only",
                data=csv_buf,
                file_name=f"REBA_scores_{timestamp}.csv",
                mime="text/csv"
            )
            st.caption("Raw scores only")
