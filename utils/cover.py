"""Small UI helpers shared by the Streamlit pages."""

##### IMPORTS ##### ------------------------------------------------------------
import base64
import html
import streamlit as st


##### PARAMETERS ##### ---------------------------------------------------------
_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 300'>
<rect width='200' height='300' fill='#3b3f46'/>
<rect x='8' y='8' width='184' height='284' fill='none' stroke='#6b7280' stroke-width='2' stroke-dasharray='6 5'/>
<g fill='none' stroke='#9ca3af' stroke-width='5' stroke-linejoin='round'>
<path d='M60 105 h40 a10 10 0 0 1 10 10 v60 a10 10 0 0 0 -10 -10 h-40 z'/>
<path d='M140 105 h-40 a10 10 0 0 0 -10 10 v60 a10 10 0 0 1 10 -10 h40 z'/></g>
<text x='100' y='215' fill='#d1d5db' font-family='sans-serif' font-size='17' text-anchor='middle'>No cover</text>
<text x='100' y='238' fill='#d1d5db' font-family='sans-serif' font-size='17' text-anchor='middle'>for this book</text>
</svg>"""
PLACEHOLDER = "data:image/svg+xml;base64," + base64.b64encode(_SVG.encode()).decode()
COVER_HEIGHT = 220      # px, same for every book on every page


##### FUNCTIONS ##### ----------------------------------------------------------
def cover_src(url):
    """The exact image URL used for display (and so for preloading)."""
    return url or None      # no -L upgrade: -M is plenty for a 220px-high cover

def cover(url, title="", height=COVER_HEIGHT):
    """
        Shows a book cover at a fixed height (2:3 ratio), centered in its
        container. Falls back to a "no cover" placeholder if there is no URL
        or the image can't be loaded.

        Parameters
        ----------
        url: str
            The url to a book's cover.
        title: str
            The title corresponding to the book's cover.
        height: int
            Height of the cover, in pixels.
            
        Returns
        -------
        A st.markdown object containing an image.
    """
    layers = f"url('{PLACEHOLDER}')"
    src = cover_src(url)
    if src:
        safe = (src.replace("'", "%27").replace('"', "%22")
                   .replace("(", "%28").replace(")", "%29"))
        layers = f"url('{safe}'), " + layers
    width = round(height * 2 / 3)
    st.markdown(
        f'<div role="img" aria-label="{html.escape(title, quote=True)}" '
        f'style="height:{height}px;width:{width}px;max-width:100%;margin:0 auto;'
        f'background-image:{layers};background-size:cover;background-position:center;'
        'background-repeat:no-repeat;border-radius:6px;'
        'box-shadow:0 1px 4px rgba(0,0,0,.35)"></div>',
        unsafe_allow_html=True)