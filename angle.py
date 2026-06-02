# ==========================================================
# angle.py — Coordinate to Angle Calculation v2
# ==========================================================

import numpy as np

# ============================================================
# CONSTANTS
# ============================================================

COCO17_NAMES = [
    "Nose","Left_Eye","Right_Eye","Left_Ear","Right_Ear",
    "Left_Shoulder","Right_Shoulder","Left_Elbow","Right_Elbow",
    "Left_Wrist","Right_Wrist","Left_Hip","Right_Hip",
    "Left_Knee","Right_Knee","Left_Ankle","Right_Ankle"
]

KP_CONF_THRESHOLD = 0.3

# ============================================================
# HELPERS
# ============================================================

def angle_2v(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    cos  = np.clip(np.dot(a, b) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

def kp_ok(kpc, i, thr=None):
    thr = thr if thr is not None else KP_CONF_THRESHOLD
    return float(kpc[i]) > thr

def kps_ok(kpc, idxs, thr=None):
    thr = thr if thr is not None else KP_CONF_THRESHOLD
    return all(float(kpc[i]) > thr for i in idxs)

# ============================================================
# 2D ANGLE COMPUTATION
# ============================================================

def compute_angles_2d(kps, kpc):

    UP   = np.array([0, -1], dtype=float)
    DOWN = np.array([0,  1], dtype=float)
    SIDE = np.array([1,  0], dtype=float)

    def v2(a, b):
        return np.array([b[0]-a[0], b[1]-a[1]], dtype=float)

    def ang3(A, B, C):
        return angle_2v(v2(B, A), v2(B, C))

    # ── Reference points ─────────────────────────────────
    if   kps_ok(kpc,[5,6]): mid_sh=(kps[5]+kps[6])/2
    elif kp_ok(kpc,5):      mid_sh=kps[5].copy()
    elif kp_ok(kpc,6):      mid_sh=kps[6].copy()
    else:                   mid_sh=None

    if   kps_ok(kpc,[11,12]): mid_hp=(kps[11]+kps[12])/2
    elif kp_ok(kpc,11):       mid_hp=kps[11].copy()
    elif kp_ok(kpc,12):       mid_hp=kps[12].copy()
    else:                     mid_hp=None

    # ── TRUNK ────────────────────────────────────────────
    if mid_sh is not None and mid_hp is not None and mid_sh[1] < mid_hp[1]:
        tv        = v2(mid_hp, mid_sh)
        trunk_fwd = float(np.clip(angle_2v(tv, UP), 0, 90))
    else:
        trunk_fwd = 0.0

    # Trunk twist — only if shoulder line CLEARLY different from hip line
    if kps_ok(kpc,[5,6,11,12]):
        sh_line = v2(kps[5],kps[6])
        hp_line = v2(kps[11],kps[12])
        twist_ang = angle_2v(sh_line, hp_line)
        # Only flag twist if > 20° (stricter) AND trunk is upright
        trunk_twisted = twist_ang > 20 and trunk_fwd < 30
    else:
        trunk_twisted = False

    # Trunk side bend
    if mid_sh is not None and mid_hp is not None and mid_sh[1] < mid_hp[1]:
        tv2 = v2(mid_hp, mid_sh)
        trunk_side_bent = (abs(tv2[0])/(abs(tv2[1])+1e-9)) > 0.35
    else:
        trunk_side_bent = False

    # ── NECK ─────────────────────────────────────────────
    if kp_ok(kpc,0,0.35) and mid_sh is not None and mid_hp is not None:
        head = kps[0]
        if head[1] < mid_sh[1]:
            nv       = v2(mid_sh, head)
            tv3      = v2(mid_hp, mid_sh)
            neck_raw = angle_2v(nv, tv3)
            neck_fwd = 180 - neck_raw
            # Sanity: standing upright + neck=70° = detection error
            if neck_fwd >= 69.0 and trunk_fwd <= 20.0:
                neck_fwd = 0.0
            else:
                neck_fwd = float(np.clip(neck_fwd, -30, 70))
        else:
            neck_fwd = 0.0
    else:
        neck_fwd = 0.0

    # Neck twist — only high confidence ears
    if kps_ok(kpc,[3,4],0.55) and kps_ok(kpc,[5,6],0.4):
        ear_line = v2(kps[3],kps[4])
        sh_line2 = v2(kps[5],kps[6])
        neck_twisted = angle_2v(ear_line,sh_line2) > 25
    else:
        neck_twisted = False

    # Neck side bend
    if kps_ok(kpc,[3,4],0.55):
        ear_line2      = v2(kps[3],kps[4])
        neck_side_bent = abs(angle_2v(ear_line2, SIDE)-90) > 20
    else:
        neck_side_bent = False

    # ── KNEE BEND ────────────────────────────────────────
    def valid_knee(hip, knee, ankle):
        return float(hip[1]) < float(knee[1]) < float(ankle[1])

    def calc_knee_bend(hip, knee, ankle):
        if not valid_knee(hip, knee, ankle):
            return 0.0
        interior  = ang3(hip, knee, ankle)
        bend_2d   = 180 - interior
        # Vertical-only estimate for side-view cross-check
        hip_y   = float(hip[1]); knee_y=float(knee[1]); ankle_y=float(ankle[1])
        vec_up  = knee_y - hip_y;   vec_down = ankle_y - knee_y
        ratio   = min(vec_up,vec_down)/(vec_up+vec_down+1e-9)
        bend_v  = ratio * 180
        # If 2D says very bent but vertical says straight → trust vertical
        if bend_2d > 90 and bend_v < 30:
            return float(np.clip(bend_v, 0, 150))
        return float(np.clip(bend_2d, 0, 150))

    r_kb=l_kb=0.0
    if kps_ok(kpc,[12,14,16]): r_kb=calc_knee_bend(kps[12],kps[14],kps[16])
    if kps_ok(kpc,[11,13,15]): l_kb=calc_knee_bend(kps[11],kps[13],kps[15])
    knee_bend = max(r_kb, l_kb)

    # ── UPPER ARM ────────────────────────────────────────
    def calc_upper_arm(shoulder, elbow):
        dx = abs(float(elbow[0]) - float(shoulder[0]))
        dy = float(elbow[1])    - float(shoulder[1])  # + = elbow below
        if dy <= 0:
            # Elbow above shoulder = arm raised
            return float(np.clip(angle_2v(v2(shoulder,elbow), DOWN), 0, 180))
        # Side-view correction
        ratio = dx / (dy + 1e-9)
        if ratio > 1.5:
            # Large horizontal shift relative to drop = side view distortion
            angle_vert = float(np.degrees(np.arctan2(dx, dy)))
            return float(np.clip(angle_vert * 0.5, 0, 45))
        return float(np.clip(angle_2v(v2(shoulder,elbow), DOWN), 0, 180))

    r_ua=l_ua=0.0
    if kps_ok(kpc,[6,8]): r_ua=calc_upper_arm(kps[6],kps[8])
    if kps_ok(kpc,[5,7]): l_ua=calc_upper_arm(kps[5],kps[7])
    upper_arm = max(r_ua, l_ua)

    upper_is_ext = False
    if kps_ok(kpc,[6,8]) and kps[8][1]<kps[6][1]: upper_is_ext=True
    if kps_ok(kpc,[5,7]) and kps[7][1]<kps[5][1]: upper_is_ext=True

    # ── SHOULDER RAISED ──────────────────────────────────
    # Only flag if one shoulder is CLEARLY higher — strict check
    if mid_sh is not None and mid_hp is not None and mid_sh[1]<mid_hp[1]:
        body_h = abs(mid_sh[1]-mid_hp[1])
        if kps_ok(kpc,[5,6]):
            sh_diff = abs(kps[5][1]-kps[6][1])
            # Require >20% of body height difference (was 15%)
            shoulder_raised = sh_diff > (body_h * 0.20)
        else:
            shoulder_raised = False
    else:
        shoulder_raised = False

    # ── ARM ABDUCTED ─────────────────────────────────────
    r_abd=l_abd=90.0
    if kps_ok(kpc,[6,8]): r_abd=angle_2v(v2(kps[6],kps[8]),SIDE)
    if kps_ok(kpc,[5,7]): l_abd=angle_2v(v2(kps[5],kps[7]),SIDE)
    # Only flag if arm is ALSO clearly raised (>30°)
    arm_abducted = (r_abd<45 or l_abd<45) and upper_arm > 30

    # ── ELBOW ────────────────────────────────────────────
    r_el=l_el=90.0
    if kps_ok(kpc,[6,8,10],0.4):
        r_el=float(np.clip(ang3(kps[6],kps[8],kps[10]),0,170))
    elif kps_ok(kpc,[6,8],0.4):
        r_el=90.0
    if kps_ok(kpc,[5,7,9],0.4):
        l_el=float(np.clip(ang3(kps[5],kps[7],kps[9]),0,170))
    elif kps_ok(kpc,[5,7],0.4):
        l_el=90.0
    elbow_ang = min(r_el, l_el)

    # ── WRIST ────────────────────────────────────────────
    r_wd=l_wd=0.0
    if kps_ok(kpc,[6,8,10],0.45):
        r_wd=float(np.clip(abs(180-ang3(kps[6],kps[8],kps[10])),0,70))
    if kps_ok(kpc,[5,7,9],0.45):
        l_wd=float(np.clip(abs(180-ang3(kps[5],kps[7],kps[9])),0,70))
    wrist_dev = max(r_wd, l_wd)
    if not kps_ok(kpc,[9,10],0.45): wrist_dev=0.0

    # Final reset: neck if nose truly not visible
    if not kp_ok(kpc,0,0.35): neck_fwd=0.0

    return dict(
        trunk_fwd=trunk_fwd,         trunk_twisted=trunk_twisted,
        trunk_side_bent=trunk_side_bent,
        neck_fwd=neck_fwd,           neck_twisted=neck_twisted,
        neck_side_bent=neck_side_bent,
        knee_bend=knee_bend,
        upper_arm_ang=upper_arm,     upper_is_ext=upper_is_ext,
        shoulder_raised=shoulder_raised, arm_abducted=arm_abducted,
        elbow_ang=elbow_ang,         wrist_dev=wrist_dev,
    )

# ============================================================
# 3D ANGLE COMPUTATION (MiDaS)
# ============================================================

def compute_angles_3d(kps, kpc, depth, cx, cy, fx, fy, w, h):
    pts=[]
    for (u,v) in kps:
        ui=int(np.clip(u,0,w-1)); vi=int(np.clip(v,0,h-1))
        Z=float(depth[vi,ui])
        pts.append([(u-cx)*Z/fx,(v-cy)*Z/fy,Z])
    pts=np.array(pts)

    UP=np.array([0,-1,0]); DOWN=np.array([0,1,0]); SIDE=np.array([1,0,0])
    def v3(a,b): return b-a
    def ang3d(A,B,C): return angle_2v(v3(B,A),v3(B,C))

    mid_sh=(pts[5]+pts[6])/2  if kps_ok(kpc,[5,6])   else None
    mid_hp=(pts[11]+pts[12])/2 if kps_ok(kpc,[11,12]) else None

    if mid_sh is not None and mid_hp is not None and mid_sh[1]<mid_hp[1]:
        tv=mid_sh-mid_hp
        trunk_fwd=float(np.clip(angle_2v(tv[:2],np.array([0,-1])),0,90))
    else:
        trunk_fwd=0.0

    if kps_ok(kpc,[5,6,11,12]):
        twist_ang=angle_2v(pts[5]-pts[6],pts[11]-pts[12])
        trunk_twisted=twist_ang>20 and trunk_fwd<30
    else:
        trunk_twisted=False

    trunk_side_bent=((abs(mid_sh[0]-mid_hp[0])/(abs(mid_sh[1]-mid_hp[1])+1e-9))>0.35
                     if (mid_sh is not None and mid_hp is not None) else False)

    if kp_ok(kpc,0,0.35) and mid_sh is not None and mid_hp is not None and pts[0][1]<mid_sh[1]:
        nv=pts[0]-mid_sh; tv3=mid_sh-mid_hp
        neck_fwd=float(180-angle_2v(nv,tv3))
        if neck_fwd>=69.0 and trunk_fwd<=20.0: neck_fwd=0.0
        else: neck_fwd=float(np.clip(neck_fwd,-30,70))
    else:
        neck_fwd=0.0

    neck_twisted   =(angle_2v(pts[3]-pts[4],pts[5]-pts[6])>25
                     if kps_ok(kpc,[3,4,5,6],0.55) else False)
    neck_side_bent =(abs(angle_2v((pts[3]-pts[4])[:2],np.array([1,0]))-90)>20
                     if kps_ok(kpc,[3,4],0.55) else False)

    def valid_k3(h,k,a): return float(h[1])<float(k[1])<float(a[1])
    def calc_kb3(h,k,a):
        if not valid_k3(h,k,a): return 0.0
        b2d=180-ang3d(h,k,a)
        vu=float(k[1])-float(h[1]); vd=float(a[1])-float(k[1])
        bv=min(vu,vd)/(vu+vd+1e-9)*180
        if b2d>90 and bv<30: return float(np.clip(bv,0,150))
        return float(np.clip(b2d,0,150))

    r_kb=calc_kb3(pts[12],pts[14],pts[16]) if kps_ok(kpc,[12,14,16]) else 0.0
    l_kb=calc_kb3(pts[11],pts[13],pts[15]) if kps_ok(kpc,[11,13,15]) else 0.0
    knee_bend=max(r_kb,l_kb)

    r_ua=float(np.clip(angle_2v(pts[8]-pts[6],DOWN),0,180)) if kps_ok(kpc,[6,8]) else 0.0
    l_ua=float(np.clip(angle_2v(pts[7]-pts[5],DOWN),0,180)) if kps_ok(kpc,[5,7]) else 0.0
    upper_arm=max(r_ua,l_ua)
    upper_is_ext=(pts[8][2]<pts[6][2]-0.05 or pts[7][2]<pts[5][2]-0.05)

    if mid_sh is not None and mid_hp is not None:
        bh=abs(mid_sh[1]-mid_hp[1])
        shoulder_raised=(abs(pts[5][1]-pts[6][1])>bh*0.20 if kps_ok(kpc,[5,6]) else False)
    else:
        shoulder_raised=False

    r_abd=angle_2v(pts[8]-pts[6],SIDE) if kps_ok(kpc,[6,8]) else 90.0
    l_abd=angle_2v(pts[7]-pts[5],SIDE) if kps_ok(kpc,[5,7]) else 90.0
    arm_abducted=(r_abd<45 or l_abd<45) and upper_arm>30

    r_el=float(np.clip(ang3d(pts[6],pts[8],pts[10]),0,170)) if kps_ok(kpc,[6,8,10],0.4) else 90.0
    l_el=float(np.clip(ang3d(pts[5],pts[7],pts[9]),0,170))  if kps_ok(kpc,[5,7,9],0.4)  else 90.0
    elbow_ang=min(r_el,l_el)

    r_wd=float(np.clip(abs(180-ang3d(pts[6],pts[8],pts[10])),0,70)) if kps_ok(kpc,[6,8,10],0.45) else 0.0
    l_wd=float(np.clip(abs(180-ang3d(pts[5],pts[7],pts[9])),0,70))  if kps_ok(kpc,[5,7,9],0.45)  else 0.0
    wrist_dev=max(r_wd,l_wd)
    if not kps_ok(kpc,[9,10],0.45): wrist_dev=0.0
    if not kp_ok(kpc,0,0.35): neck_fwd=0.0

    return dict(
        trunk_fwd=trunk_fwd,         trunk_twisted=trunk_twisted,
        trunk_side_bent=trunk_side_bent,
        neck_fwd=neck_fwd,           neck_twisted=neck_twisted,
        neck_side_bent=neck_side_bent,
        knee_bend=knee_bend,
        upper_arm_ang=upper_arm,     upper_is_ext=upper_is_ext,
        shoulder_raised=shoulder_raised, arm_abducted=arm_abducted,
        elbow_ang=elbow_ang,         wrist_dev=wrist_dev,
    )
