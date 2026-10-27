"""Run it from terminal with: $ streamlit run app.py"""

##### IMPORTS ##### ------------------------------------------------------------
import streamlit as st


##### PARAMETERS ##### ---------------------------------------------------------
st.set_page_config(
    page_title="Babl", 
    page_icon="./static/images/logo.png", 
    layout="wide",
    initial_sidebar_state="expanded")
st.logo(
    "./static/images/logo.png",
    icon_image="./static/images/logo.png"
)


##### APP ##### ----------------------------------------------------------------
page = st.navigation([
    st.Page(
        "pages/rating.py", 
        title="Annotation", 
        icon=":material/edit_note:", 
        default=True),
    st.Page(
        "pages/recommend.py", 
        title="Recommendation", 
        icon=":material/editor_choice:"),
    st.Page(
        "pages/masse_critique.py", 
        title="Masse Critique (Babelio)", 
        icon=":material/event_upcoming:"),
])

page.run()