# panel4_reba.py

import streamlit as st


def get_action(reba_score):

    if reba_score == 1:
        return "No action required"

    elif reba_score <= 3:
        return "May need action"

    elif reba_score <= 7:
        return "Further investigate. Change soon."

    elif reba_score <= 10:
        return "Investigate and implement change"

    else:
        return "Implement change immediately"


def get_risk_color(reba_score):

    if reba_score == 1:
        return "#4CAF50"      # Green

    elif reba_score <= 3:
        return "#8BC34A"      # Light Green

    elif reba_score <= 7:
        return "#FF9800"      # Orange

    elif reba_score <= 10:
        return "#F44336"      # Red

    else:
        return "#8B0000"      # Dark Red


def render_panel4(
    neck,
    trunk,
    leg,
    upper_arm,
    lower_arm,
    wrist,
    table_a,
    load_force,
    score_a,
    table_b,
    coupling,
    score_b,
    table_c,
    activity,
    reba_score,
    risk_level
):

    st.markdown("""
    <style>

    .panel-header{
        background:#1E293B;
        padding:12px;
        border-radius:10px;
        margin-bottom:10px;
        color:white;
        font-weight:bold;
        font-size:20px;
    }

    .workflow-box{
        border:1px solid #CBD5E1;
        border-radius:8px;
        padding:10px;
        text-align:center;
        background:white;
        margin-bottom:8px;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="panel-header">4. REBA ASSESSMENT WORKFLOW</div>',
        unsafe_allow_html=True
    )

    colA, colB, colC = st.columns([1, 1, 1])

    # ==================================
    # GROUP A
    # ==================================

    with colA:

        st.markdown("### Group A")
        st.markdown(
            f"""
            <div class="workflow-box">
            Neck : <b>{neck}</b><br>
            Trunk : <b>{trunk}</b><br>
            Leg : <b>{leg}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="workflow-box">
            Table A<br>
            <h3>{table_a}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="workflow-box">
            Load / Force Score<br>
            <h3>{load_force}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="workflow-box">
            Score A<br>
            <h3>{score_a}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==================================
    # GROUP B
    # ==================================

    with colB:

        st.markdown("### Group B")

        st.markdown(
            f"""
            <div class="workflow-box">
            Upper Arm : <b>{upper_arm}</b><br>
            Lower Arm : <b>{lower_arm}</b><br>
            Wrist : <b>{wrist}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="workflow-box">
            Table B<br>
            <h3>{table_b}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="workflow-box">
            Coupling Score<br>
            <h3>{coupling}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="workflow-box">
            Score B<br>
            <h3>{score_b}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ==================================
    # FINAL RESULT
    # ==================================

    with colC:

        st.markdown(
            f"""
            <div class="workflow-box">
            Table C<br>
            <h3>{table_c}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="workflow-box">
            Activity Score<br>
            <h3>{activity}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="workflow-box">
            Final REBA Score<br>
            <h2>{reba_score}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        color = get_risk_color(reba_score)

        st.markdown(
            f"""
            <div style="
                background:{color};
                padding:12px;
                border-radius:8px;
                text-align:center;
                color:white;
                font-weight:bold;
                margin-bottom:8px;
            ">
            Ergonomic Risk Level<br>
            {risk_level}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="workflow-box">
            Recommended Action<br><br>
            <b>{get_action(reba_score)}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### REBA Scoring Guide")

    st.markdown("""
    🟢 **1** → Negligible Risk

    🟢 **2–3** → Low Risk

    🟠 **4–7** → Medium Risk

    🔴 **8–10** → High Risk

    🟥 **11+** → Very High Risk
    """)