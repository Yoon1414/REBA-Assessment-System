# ==========================================================
# reba.py — REBA Tables + Scoring Logic
# ==========================================================

# ============================================================
# REBA LOOKUP TABLES
# ============================================================

_A1 = {
    1:{1:[1,2,3,4],2:[1,2,3,3],3:[3,3,5,6]},
    2:{1:[2,3,4,5],2:[1,2,3,4],3:[5,6,6,7]},
    3:{1:[2,4,5,6],2:[4,5,6,7],3:[5,6,7,8]},
    4:{1:[3,5,6,7],2:[5,6,7,8],3:[6,7,8,9]},
    5:{1:[4,6,7,8],2:[6,7,8,9],3:[7,8,9,9]},
}

_B1 = {
    1:{1:{1:1,2:2,3:2},2:{1:1,2:2,3:3}},
    2:{1:{1:1,2:2,3:3},2:{1:2,2:3,3:4}},
    3:{1:{1:3,2:4,3:5},2:{1:4,2:5,3:5}},
    4:{1:{1:4,2:5,3:5},2:{1:5,2:6,3:7}},
    5:{1:{1:6,2:7,3:8},2:{1:7,2:8,3:8}},
    6:{1:{1:7,2:8,3:8},2:{1:8,2:9,3:9}},
}

_C = [
    [1,2,3,3,4,5,6,7,7,8,8,9],
    [2,3,4,4,5,6,7,8,8,9,9,9],
    [3,4,5,5,6,7,8,8,9,9,9,9],
    [4,5,6,6,7,8,8,9,9,9,9,9],
    [5,6,7,7,8,8,9,9,9,9,9,9],
    [6,7,8,8,8,9,9,9,9,9,9,9],
    [7,8,8,8,9,9,9,9,9,9,9,9],
    [8,9,9,9,10,10,10,10,10,10,10,10],
    [9,10,10,10,11,11,11,11,11,11,11,11],
    [10,11,11,11,12,12,12,12,12,12,12,12],
    [11,12,12,12,12,12,12,12,12,12,12,12],
    [12,12,12,12,12,12,12,12,12,12,12,12],
]

# ============================================================
# HELPERS
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def risk_level(s):
    """Convert REBA score to risk label."""
    if   s == 1:        return "Negligible"
    elif 2 <= s <= 3:   return "Low"
    elif 4 <= s <= 7:   return "Medium"
    elif 8 <= s <= 10:  return "High"
    else:               return "Very High"

def risk_color_bgr(s):
    """Return BGR color for risk level (for OpenCV drawing)."""
    if   s == 1:        return (0, 200, 0)      # green
    elif 2 <= s <= 3:   return (0, 220, 220)    # yellow
    elif 4 <= s <= 7:   return (0, 140, 255)    # orange
    elif 8 <= s <= 10:  return (0, 0, 255)      # red
    else:               return (0, 0, 160)      # dark red

# ============================================================
# REBA SCORING
# Official worksheet — exact degrees
# ============================================================

def compute_reba(angles,
                 one_leg_raised=False,
                 force_load=0,
                 shock=False,
                 arm_supported=False,
                 coupling=0,
                 wrist_twisted=False,
                 act_static=False,
                 act_repeated=False,
                 act_rapid=False):
    """
    Compute full REBA score from angles dict.

    Parameters
    ----------
    angles        : dict from angle.compute_angles_2d() or compute_angles_3d()
    one_leg_raised: bool   — Step 3 leg base (+1)
    force_load    : 0/1/2  — Step 5 load score
    shock         : bool   — Step 5 shock/rapid force (+1)
    arm_supported : bool   — Step 7a arm supported (-1)
    coupling      : 0/1/2/3— Step 11 grip quality
    wrist_twisted : bool   — Step 9a wrist twist (+1)
    act_static    : bool   — Step 13 static >1min (+1)
    act_repeated  : bool   — Step 13 repeated >4x/min (+1)
    act_rapid     : bool   — Step 13 rapid changes (+1)

    Returns
    -------
    dict with all bins, intermediate scores, and final REBA score
    """

    # ── STEP 1: NECK ─────────────────────────────────────
    # +1 = 0–20° forward
    # +2 = >20° forward OR any extension (negative)
    nf = angles["neck_fwd"]
    nb = 2 if (nf < 0 or nf > 20) else 1
    neck_adj   = int(angles["neck_twisted"]) + int(angles["neck_side_bent"])
    neck_final = clamp(nb + neck_adj, 1, 3)

    # ── STEP 2: TRUNK ────────────────────────────────────
    # +1 = 0–5°  (nearly upright)
    # +2 = 5–20° forward OR 0–20° extension
    # +3 = 20–60° forward OR >20° extension
    # +4 = >60° forward
    tf = angles["trunk_fwd"]
    if   tf < 0:    tb = 2 if abs(tf) <= 20 else 3
    elif tf <= 5:   tb = 1
    elif tf <= 20:  tb = 2
    elif tf <= 60:  tb = 3
    else:           tb = 4
    trunk_adj   = int(angles["trunk_twisted"]) + int(angles["trunk_side_bent"])
    trunk_final = clamp(tb + trunk_adj, 1, 5)

    # ── STEP 3: LEGS ─────────────────────────────────────
    # base: +1 both legs down, +2 one leg raised
    # adjust: +1 if knee 30–60°, +2 if knee >60°
    base_leg = 2 if one_leg_raised else 1
    kb       = angles["knee_bend"]
    kadj     = 0 if kb < 30 else (1 if kb <= 60 else 2)
    leg_final = clamp(base_leg + kadj, 1, 4)

    # ── STEP 4: TABLE A ──────────────────────────────────
    A_pos = _A1[trunk_final][neck_final][leg_final - 1]

    # ── STEP 5: FORCE / LOAD ─────────────────────────────
    force_sc = clamp(force_load + int(shock), 0, 3)

    # ── STEP 6: SCORE A = Table A + Force ────────────────
    Score_A = clamp(A_pos + force_sc, 1, 12)

    # ── STEP 7: UPPER ARM ────────────────────────────────
    # +1 = ±20° (arm by side)
    # +2 = 20–45° forward OR >20° extension
    # +3 = 45–90° forward
    # +4 = >90° forward (arm raised above shoulder)
    ua = angles["upper_arm_ang"]
    if angles["upper_is_ext"]:
        ub = 1 if ua <= 20 else 2
    else:
        if   ua <= 20: ub = 1
        elif ua <= 45: ub = 2
        elif ua <= 90: ub = 3
        else:          ub = 4
    upper_adj   = (int(angles["shoulder_raised"]) +
                   int(angles["arm_abducted"]) -
                   int(arm_supported))
    upper_final = clamp(ub + upper_adj, 1, 6)

    # ── STEP 8: LOWER ARM ────────────────────────────────
    # +1 = 60–100° (good working range)
    # +2 = outside range
    ea = angles["elbow_ang"]
    lb = 1 if 60 <= ea <= 100 else 2
    lower_final = clamp(lb, 1, 2)

    # ── STEP 9: WRIST ────────────────────────────────────
    # +1 = ±15° neutral
    # +2 = >15° bent
    # +1 extra if twisted
    wd = angles["wrist_dev"]
    wb = 1 if wd <= 15 else 2
    if wrist_twisted: wb += 1
    wrist_final = clamp(wb, 1, 3)

    # ── STEP 10: TABLE B ─────────────────────────────────
    B_pos = _B1[upper_final][lower_final][wrist_final]

    # ── STEP 11: COUPLING ────────────────────────────────
    coup_sc = clamp(coupling, 0, 3)

    # ── STEP 12: SCORE B = Table B + Coupling ────────────
    Score_B = clamp(B_pos + coup_sc, 1, 12)

    # ── TABLE C ──────────────────────────────────────────
    C_score = _C[Score_A - 1][Score_B - 1]

    # ── STEP 13: ACTIVITY ────────────────────────────────
    activity = int(act_static) + int(act_repeated) + int(act_rapid)

    # ── FINAL REBA ───────────────────────────────────────
    REBA = C_score + activity

    return dict(
        # Raw angles
        trunk_fwd      = round(angles["trunk_fwd"], 1),
        neck_fwd       = round(angles["neck_fwd"], 1),
        knee_bend      = round(angles["knee_bend"], 1),
        upper_arm      = round(angles["upper_arm_ang"], 1),
        elbow          = round(angles["elbow_ang"], 1),
        wrist_dev      = round(angles["wrist_dev"], 1),
        # Flags
        trunk_twisted  = angles["trunk_twisted"],
        neck_twisted   = angles["neck_twisted"],
        shoulder_raised= angles["shoulder_raised"],
        arm_abducted   = angles["arm_abducted"],
        # Bins — raw / adjustment / final
        trunk_bin_raw  = tb,    trunk_adj = trunk_adj, trunk_bin = trunk_final,
        neck_bin_raw   = nb,    neck_adj  = neck_adj,  neck_bin  = neck_final,
        leg_bin        = leg_final,
        upper_bin_raw  = ub,    upper_adj = upper_adj, upper_bin = upper_final,
        lower_bin      = lower_final,
        wrist_bin      = wrist_final,
        # Scores
        A_posture      = A_pos,
        force_score    = force_sc,
        Score_A        = Score_A,
        B_posture      = B_pos,
        coupling       = coup_sc,
        Score_B        = Score_B,
        C_score        = C_score,
        activity       = activity,
        REBA_final     = REBA,
        risk           = risk_level(REBA),
    )
