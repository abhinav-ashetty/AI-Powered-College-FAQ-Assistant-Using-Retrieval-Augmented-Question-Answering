import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="College Academic FAQ Assistant", layout="wide")

st.title("🎓 College Academic FAQ Assistant")
st.caption("Ask questions about course syllabus, CIE/SEE marking schemes, attendance, and exam rules.")

# Sidebar Filters
st.sidebar.header("Academic Filters")
department = st.sidebar.selectbox(
    "Department",
    ["All", "ISE", "CSE", "ECE", "MECH", "CIVIL", "General"],
    index=1  # Default to ISE
)

semester = st.sidebar.selectbox(
    "Semester",
    ["All", "Sem 1", "Sem 2", "Sem 3", "Sem 4", "Sem 5", "Sem 6", "Sem 7", "Sem 8"],
    index=7  # Default to Sem 7
)

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📚 Sources: {', '.join(msg['sources'])}")

# User query input
if prompt := st.chat_input("Ask a question (e.g., What are the passing marks for BIS701?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "question": prompt,
        "department": department,
        "semester": semester
    }

    with st.chat_message("assistant"):
        with st.spinner("Retrieving syllabus regulations and generating answer..."):
            try:
                response = requests.post(
                    f"{API_BASE}/api/ask",
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        st.caption(f"📚 Sources: {', '.join(sources)}")
                        
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    err_msg = f"API Error ({response.status_code}): {response.text}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

            except requests.exceptions.ReadTimeout:
                err_msg = "⏱️ The backend took longer than 120 seconds to respond. Please try asking again."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except requests.exceptions.ConnectionError:
                err_msg = "🔌 Could not connect to FastAPI server. Please check if Uvicorn is running on port 8000."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as e:
                err_msg = f"Unexpected error: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
