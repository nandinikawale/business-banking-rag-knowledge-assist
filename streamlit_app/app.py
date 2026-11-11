import streamlit as st

from banking_rag.models import QueryRequest
from banking_rag.pipeline import RAGPipeline

st.set_page_config(page_title="Business Banking Knowledge Assistant", page_icon="📚", layout="wide")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline()
    pipeline.ensure_index()
    return pipeline


st.title("Business Banking RAG Knowledge Assistant")
st.caption("Cited answers from an entirely fictional policy and product knowledge base")
st.info(
    "Portfolio demonstration only. The content is fictional and is not legal, compliance, "
    "credit, or product advice. Do not use it for real customer decisions."
)

with st.sidebar:
    st.header("Retrieval settings")
    category_label = st.selectbox(
        "Knowledge category",
        ["All", "Lending", "Deposits", "Treasury", "Customer due diligence", "RM playbook"],
    )
    top_k = st.slider("Sources to retrieve", 2, 6, 4)
    st.divider()
    st.markdown("**Architecture**")
    st.write("Markdown → chunks → local embeddings → Chroma → grounded synthesis")

examples = [
    "What documents are needed for a business line of credit request?",
    "When should an RM escalate a sustained deposit decline?",
    "What should be reviewed before discussing ACH services?",
]
question = st.text_area(
    "Ask a policy, product, or RM-process question", value=examples[0], height=100
)

if st.button("Retrieve grounded answer", type="primary", use_container_width=True):
    category = None if category_label == "All" else category_label
    with st.spinner("Searching the fictional knowledge base..."):
        response = get_pipeline().answer(
            QueryRequest(question=question, top_k=top_k, category=category)
        )
    st.subheader("Answer")
    st.markdown(response.answer)
    st.caption(f"Mode: {response.mode} · Supported: {'Yes' if response.supported else 'No'}")
    st.subheader("Sources")
    if not response.sources:
        st.warning("No sufficiently relevant source was found.")
    for index, source in enumerate(response.sources, start=1):
        with st.expander(f"[{index}] {source.title} — {source.section}", expanded=index == 1):
            st.write(source.text)
            st.caption(
                f"Document: {source.document_id} · Category: {source.category} · "
                f"Version: {source.version} · Relevance: {source.relevance:.2f}"
            )
