"""
Streamlit Chat UI for the FAQ Knowledge Base Assistant.

Uses the StreamlitAdapter to call the RAG engine directly (no HTTP overhead).
For production deployment, switch to the HTTP API endpoint instead.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.core.rag_engine import get_rag_engine
from src.adapters.streamlit_adapter import StreamlitAdapter

# Page config
st.set_page_config(
    page_title="AI 知识库助手",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage { padding: 1rem; }
    .source-box {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 10px;
        margin-top: 10px;
        font-size: 0.85em;
    }
    .counter-q {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        border-radius: 4px;
        margin: 10px 0;
    }
    .stButton button {
        width: 100%;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 内部AI知识库助手")
st.caption("覆盖出差注意事项 | 马来西亚商务签证 | 项目申报材料")

# Initialize the adapter (singleton, created once)
@st.cache_resource
def get_adapter() -> StreamlitAdapter:
    engine = get_rag_engine()
    engine.initialize()
    return StreamlitAdapter(engine)

adapter = get_adapter()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "您好！我是内部知识库助手。\n\n我可以帮您解答以下方面的问题：\n"
                       "- 🌴 出差注意事项（物品准备、安全、文化禁忌）\n"
                       "- 🛂 马来西亚商务签证办理（材料、流程、时间周期）\n"
                       "- 📋 项目申报材料（政府文件、表单填写、提交要求）\n\n"
                       "请直接提问吧！",
            "sources": [],
        }
    ]

if "pending_options" not in st.session_state:
    st.session_state.pending_options = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

# Sidebar
with st.sidebar:
    st.header("⚙️ 设置")
    domain_filter = st.selectbox(
        "知识域筛选",
        ["全部", "出差注意事项", "马来西亚商务签证", "项目申报材料"],
        index=0,
    )
    st.divider()
    st.caption("模型：Qwen2.5:3B (Ollama)")
    st.caption("检索：BM25 + Dense + RRF + Rerank")
    st.caption("版本：2.0.0")
    st.caption("模式：本地直接调用 (Adapter)")

# Domain filter mapping
domain_map = {
    "出差注意事项": "travel_tips",
    "马来西亚商务签证": "malaysia_visa",
    "项目申报材料": "project_applications",
}

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources if available
        if msg.get("sources"):
            with st.expander("📎 参考来源"):
                for j, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**{j}. {src.get('title', '未知')}** "
                        f"_(相关度: {src.get('score', src.get('relevance_score', 0)):.3f})_\n\n"
                        f"{src.get('content_snippet', src.get('content', ''))[:200]}..."
                    )

# Handle counter-question buttons
if st.session_state.pending_options:
    st.info("请选择一个方向：")
    cols = st.columns(len(st.session_state.pending_options))
    for idx, opt in enumerate(st.session_state.pending_options):
        with cols[idx]:
            if st.button(opt["label"], key=f"opt_{idx}"):
                with st.spinner("搜索中..."):
                    result = adapter.process_clarify(
                        st.session_state.pending_query,
                        opt["domain"],
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["text"],
                        "sources": result.get("sources", []),
                    })
                    st.session_state.pending_options = []
                    st.session_state.pending_query = ""
                    st.rerun()

# Chat input
if prompt := st.chat_input("输入您的问题..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare domain filter
    domain_val = domain_map.get(domain_filter) if domain_filter != "全部" else None

    # Call engine via adapter
    with st.spinner("检索知识库中..."):
        try:
            result = adapter.process_with_domain(prompt, domain_filter=domain_val)

            if result.get("action") == "counter_question":
                # Show counter-question and present options
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❓ {result['counter_question']}",
                    "sources": [],
                })
                st.session_state.pending_options = result.get("options", [])
                st.session_state.pending_query = prompt
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["text"],
                    "sources": result.get("sources", []),
                })
                st.session_state.pending_options = []
                st.session_state.pending_query = ""
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ 出错了: {str(e)}",
                "sources": [],
            })

    st.rerun()
