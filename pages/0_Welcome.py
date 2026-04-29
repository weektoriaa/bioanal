import streamlit as st

st.set_page_config(page_title="Welcome", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 2.8rem 2.9rem 2.4rem 2.9rem;
        border: 1px solid rgba(223, 136, 170, 0.32);
        background:
            linear-gradient(140deg, rgba(255, 233, 241, 0.92), rgba(255, 248, 252, 0.98)),
            linear-gradient(180deg, #ffffff, #fff7fb);
        border-radius: 16px;
        margin-bottom: 1.3rem;
        position: relative;
        overflow: hidden;
    }
    .hero:before {
        content: "🎀";
        position: absolute;
        right: 1.2rem;
        top: 0.7rem;
        font-size: 2rem;
        opacity: 0.75;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.55rem;
        line-height: 1.02;
        color: #3a2530;
        letter-spacing: 0;
    }
    .hero p {
        margin-top: 0.8rem;
        max-width: 49rem;
        color: #6d4f5b;
        font-size: 1.03rem;
    }
    .mini-note {
        color: #8a6273;
        font-size: 0.95rem;
        margin-top: 0.35rem;
    }
    .module-band {
        padding: 1.05rem 1.1rem;
        border: 1px solid rgba(223, 136, 170, 0.22);
        border-radius: 10px;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(255, 245, 250, 0.88));
        min-height: 164px;
    }
    .module-band h3 {
        margin-top: 0;
        margin-bottom: 0.45rem;
        font-size: 1.08rem;
        color: #432d37;
    }
    .module-band p {
        color: #685761;
        font-size: 0.95rem;
        margin-bottom: 0;
    }
    .soft-tag {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: rgba(212, 83, 126, 0.1);
        color: #934766;
        font-size: 0.82rem;
        margin-bottom: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="soft-tag">scientific workspace</div>
        <h1>Vikusya Bioanalysis</h1>
        <p>
            A calm pink home for biological data work: FLIM now, with room for cytometry,
            qPCR, sequencing, and shared result workflows as the platform grows.
        </p>
        <div class="mini-note">Minimal on purpose, with woman energy.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_col, launch_col = st.columns([1.35, 1])

with overview_col:
    st.subheader("What lives here")
    module_cols = st.columns(2)
    modules = [
        ("FLIM", "Upload ASC files, clean values, run statistics, and export article-friendly charts."),
        ("Flow Cytometry", "Future space for gating summaries, comparisons, and result export."),
        ("qPCR", "Future space for Ct tables, normalization, fold-change workflows, and plotting."),
        ("Sequencing", "Future space for QC summaries, derived tables, and downstream visuals."),
    ]
    for index, (title, description) in enumerate(modules):
        with module_cols[index % 2]:
            st.markdown(
                f"""
                <div class="module-band">
                    <h3>{title}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

with launch_col:
    st.subheader("Open a workspace")
    st.page_link("pages/1_FLIM.py", label="FLIM", icon=":material/scatter_plot:")
    st.page_link("pages/2_Flow_Cytometry.py", label="Flow Cytometry", icon=":material/biotech:")
    st.page_link("pages/3_qPCR.py", label="qPCR", icon=":material/functions:")
    st.page_link("pages/4_Sequencing.py", label="Sequencing", icon=":material/genetics:")
    st.page_link("pages/5_Results.py", label="Results", icon=":material/description:")
    st.caption("The FLIM module is currently the active production workspace.")

st.divider()

detail_col_1, detail_col_2, detail_col_3 = st.columns(3)

with detail_col_1:
    st.markdown("**Current strengths**")
    st.write("Custom export names, optional stats sheets, multiple plot types, pairwise brackets, and PDF output.")

with detail_col_2:
    st.markdown("**Author**")
    st.write("Monicheva Viktoria Stanislavovna")

with detail_col_3:
    st.markdown("**Next layer**")
    st.write("Split shared logic into modules and let each analysis page grow without tangling the rest of the platform.")
