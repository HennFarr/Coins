# Welcome Page
import streamlit as slit

slit.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

slit.write("# Welcome to our Project! 👋")

slit.sidebar.header("Select a Page above.")

slit.markdown(
    """
    Im Rahmen unseres Studiums haben wir (Henning Pfarren und Kim Rautenberg) an dem Projekt Data Science & MLOps teilgenommen. 

    Hier sehen Sie auf den Folgenden Seiten, das Ergebnis.

    - Auf der Seite "One Coin" können Sie anhand von einem Bild von einer Münze berechnen lassen welchen Wert diese hat.
    - Auf der Seite 'Multiple Coin" können Sie anhand von einem Bild von mehreren Münzen berechnen lassen welchen Wert diese haben.

    Viel Spaß beim ausprobieren!

"""
)