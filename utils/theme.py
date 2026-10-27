"""A function that imports colors from streamlit's theme in its .toml file."""

##### IMPORTS ##### ------------------------------------------------------------
import streamlit as st


##### FUNCTIONS ##### ----------------------------------------------------------
def theme_color(name: str, default: str = "#4c78a8") -> str:
    """Return a color from config.toml for the active light/dark theme."""
    mode = getattr(st.context, "theme", None)
    mode = (mode.type if mode else None) or "light"   # "light" or "dark"
    return (st.get_option(f"theme.{mode}.{name}")
            or st.get_option(f"theme.{name}")
            or default)