# panel1_skeleton.py

import streamlit as st


def render_panel1(
    annotated_img,
    person_id=None,
    confidence=None,
    reba_score=None,
    risk_level=None
):
    """
    Panel 1 - Skeleton Detection

    Parameters
    ----------
    annotated_img
        Image already annotated with skeleton overlay

    person_id : int
        Worker ID

    confidence : float
        YOLO detection confidence

    reba_score : int
        Final REBA score

    risk_level : str
        REBA risk level
    """

    # =====================================
    # Custom CSS
    # =====================================

    st.markdown(
        """
        <style>

        .panel-box{
            background-color:#1E293B;
            padding:12px;
            border-radius:10px;
            border:1px solid #334155;
            margin-bottom:10px;
        }

        .panel-title{
            color:white;
            font-size:20px;
            font-weight:bold;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # =====================================
    # Header
    # =====================================

    st.markdown(
        """
        <div class="panel-box">
            <div class="panel-title">
                🦴 1. SKELETON DETECTION
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================
    # Skeleton Image
    # =====================================

    st.image(
        annotated_img,
        channels="BGR",
        use_container_width=True
    )

    # =====================================
    # Information Cards
    # =====================================

    col1, col2 = st.columns(2)

    with col1:

        if person_id is not None:
            st.metric(
                label="Worker ID",
                value=str(person_id)
            )

        if confidence is not None:
            st.metric(
                label="Detection Confidence",
                value=f"{confidence:.3f}"
            )

    with col2:

        if reba_score is not None:
            st.metric(
                label="REBA Score",
                value=str(reba_score)
            )

        if risk_level is not None:

            if str(risk_level).lower().startswith("negligible"):
                st.success(f"Risk Level: {risk_level}")

            elif str(risk_level).lower().startswith("low"):
                st.info(f"Risk Level: {risk_level}")

            elif str(risk_level).lower().startswith("medium"):
                st.warning(f"Risk Level: {risk_level}")

            else:
                st.error(f"Risk Level: {risk_level}")

    # =====================================
    # Description
    # =====================================

    st.info(
        "This panel displays the YOLO11m-Pose detection results, "
        "including the worker bounding box, confidence score, "
        "17 body keypoints, and skeleton connections."
    )