# ==========================================================
# visualization.py — Skeleton Drawing + Labels
# ==========================================================

import cv2
import numpy as np

# ============================================================
# SKELETON CONFIGURATION
# ============================================================

# COCO17 keypoint connections
SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),    # face
    (5,6),                       # shoulder bar
    (5,7),(7,9),                 # left arm
    (6,8),(8,10),                # right arm
    (5,11),(6,12),               # torso sides (shoulder → hip)
    (11,12),                     # hip bar
    (11,13),(13,15),             # left leg
    (12,14),(14,16),             # right leg
]

# BGR colors per edge group
EDGE_COLORS = {
    # Face — yellow
    (0,1):(0,255,255),(0,2):(0,255,255),
    (1,3):(0,255,255),(2,4):(0,255,255),
    # Shoulder bar — white
    (5,6):(255,255,255),
    # Left arm — green
    (5,7):(0,255,0),(7,9):(0,255,0),
    # Right arm — green
    (6,8):(0,255,0),(8,10):(0,255,0),
    # Torso — cyan (most important for REBA)
    (5,11):(255,255,0),(6,12):(255,255,0),(11,12):(255,255,0),
    # Left leg — orange
    (11,13):(0,165,255),(13,15):(0,165,255),
    # Right leg — orange
    (12,14):(0,165,255),(14,16):(0,165,255),
}

# ============================================================
# DRAWING FUNCTIONS
# ============================================================

def draw_skeleton(frame, kps, kpc, kp_conf_threshold=0.3):
    """
    Draw skeleton lines and joint dots on frame.

    Parameters
    ----------
    frame             : BGR image (numpy array)
    kps               : (17,2) keypoint pixel coordinates
    kpc               : (17,) keypoint confidence scores
    kp_conf_threshold : minimum confidence to draw a joint

    Returns
    -------
    frame with skeleton drawn
    """
    # Draw connecting lines
    for (i, j) in SKELETON_EDGES:
        if i < len(kps) and j < len(kps):
            if (float(kpc[i]) > kp_conf_threshold and
                    float(kpc[j]) > kp_conf_threshold):
                p1 = (int(kps[i][0]), int(kps[i][1]))
                p2 = (int(kps[j][0]), int(kps[j][1]))
                color = EDGE_COLORS.get((i, j), (0, 255, 0))
                cv2.line(frame, p1, p2, color, 2)

    # Draw joint dots
    for idx, (x, y) in enumerate(kps):
        if float(kpc[idx]) > kp_conf_threshold:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

    return frame


def draw_label(frame, text, x1, y1, color):
    """
    Draw filled label box ABOVE bounding box.
    Shows: ID | REBA score | Risk level

    Parameters
    ----------
    frame : BGR image
    text  : label string e.g. "ID 1 | REBA 8 | High"
    x1,y1 : top-left corner of bounding box
    color : BGR color tuple

    Returns
    -------
    frame with label drawn
    """
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.6
    thick  = 2
    (tw, th), _ = cv2.getTextSize(text, font, fscale, thick)
    lx1 = int(x1)
    ly1 = max(int(y1) - th - 10, 0)
    lx2 = lx1 + tw + 8
    ly2 = int(y1)
    # Background rectangle
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
    # Text
    cv2.putText(frame, text, (lx1+4, ly2-4), font, fscale, (255,255,255), thick)
    return frame


def draw_conf_label(frame, conf, x1, y2, color):
    """
    Draw person confidence score BELOW bounding box.
    Shows: "person 0.94" style like YOLO default output.

    Parameters
    ----------
    frame   : BGR image
    conf    : float confidence score (0–1)
    x1, y2  : bottom-left corner of bounding box
    color   : BGR color tuple

    Returns
    -------
    frame with confidence label drawn
    """
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55
    thick  = 2
    text   = f"person {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, font, fscale, thick)
    lx1 = int(x1)
    ly1 = int(y2)
    lx2 = lx1 + tw + 8
    ly2 = ly1 + th + 8
    # Background rectangle
    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, -1)
    # Text
    cv2.putText(frame, text, (lx1+4, ly2-4), font, fscale, (255,255,255), thick)
    return frame


def draw_person(frame, kps, kpc, boxes_row, tid, conf, reba_result,
                kp_conf_threshold=0.3):
    """
    Draw everything for one person:
    - Skeleton
    - Bounding box
    - Top label (ID | REBA | Risk)
    - Bottom label (person confidence)

    Parameters
    ----------
    frame           : BGR image
    kps             : (17,2) keypoints
    kpc             : (17,) confidences
    boxes_row       : [x1,y1,x2,y2] bounding box
    tid             : track ID (int)
    conf            : person detection confidence
    reba_result     : dict from reba.compute_reba()

    Returns
    -------
    annotated frame
    """
    from reba import risk_color_bgr

    R     = reba_result["REBA_final"]
    color = risk_color_bgr(R)
    risk  = reba_result["risk"]

    # Skeleton
    frame = draw_skeleton(frame, kps, kpc, kp_conf_threshold)

    # Bounding box
    x1, y1, x2, y2 = [int(v) for v in boxes_row]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Top label
    label = f"ID {tid} | REBA {R} | {risk}"
    frame = draw_label(frame, label, x1, y1, color)

    # Bottom label
    frame = draw_conf_label(frame, conf, x1, y2, color)

    return frame
