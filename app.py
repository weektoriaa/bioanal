import streamlit as st

pages = {
    "Workspace": [
        st.Page("pages/0_Welcome.py", title="Welcome", icon=":material/waving_hand:"),
        st.Page("pages/1_FLIM.py", title="FLIM", icon=":material/scatter_plot:"),
        st.Page("pages/2_Flow_Cytometry.py", title="Flow Cytometry", icon=":material/biotech:"),
        st.Page("pages/3_qPCR.py", title="qPCR", icon=":material/functions:"),
        st.Page("pages/4_Sequencing.py", title="Sequencing", icon=":material/genetics:"),
        st.Page("pages/5_Results.py", title="Results", icon=":material/description:"),
    ]
}

navigation = st.navigation(pages)
navigation.run()
