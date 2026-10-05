"""The presentation app: the project, live, in talk order.

    .venv/bin/streamlit run app/streamlit_app.py
    # or with the stack:  docker compose up -d   ->  http://localhost:8501

Pages live in `fraud.dashboard.pages`; numbers in `fraud.dashboard.data`.
"""

import streamlit as st

st.set_page_config(page_title="Card Fraud MLOps", page_icon=":material/shield:", layout="wide")

from fraud.dashboard.pages import (  # noqa: E402  (set_page_config must come first)
    page_answer,
    page_monitoring,
    page_ops,
    page_replay,
    page_retrain,
    page_serving,
    page_transaction,
    sidebar,
)

# --- navigation ------------------------------------------------------------------------

PAGES = [  # (page, title, icon, url_path)
    (page_answer, "The answer", "target", ""),
    (page_replay, "Replay an evening", "play_circle", "replay"),
    (page_transaction, "One transaction", "search", "transaction"),
    (page_ops, "Registry & operations", "swap_horiz", "operations"),
    (page_monitoring, "Monitoring", "monitoring", "monitoring"),
    (page_serving, "Serving", "speed", "serving"),
    (page_retrain, "The retrain", "science", "retrain"),
]

sidebar()
st.navigation([
    st.Page(fn, title=title, icon=f":material/{icon}:", default=not path,
            **({"url_path": path} if path else {}))
    for fn, title, icon, path in PAGES
]).run()
