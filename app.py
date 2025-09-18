import streamlit as st
from rag_utils import get_yt_transcript, generate_answer 

#key setup for streamlit secrets management
api_key = st.secrets["openai"]["OPENAI_API_KEY"]



st.set_page_config(page_title="YouTube Q&A", page_icon="🎬", layout="centered")

st.title("🎬 YouTube Transcript Q&A")


# --- Input widgets ---

url = st.sidebar.text_input("YouTube Video URL:")

placeholder = st.empty()

process_url_button = st.sidebar.button("Process URL")

if process_url_button:
    if not url:
        st.sidebar.warning("You must provide the url.")

    else: 
        for status in get_yt_transcript(url):
            placeholder.status(status)


query  = placeholder.text_area("Question:")

query_button = st.button("Enter")

if query_button:           
    try:
        with st.spinner("Thinking... for better response."):
            answer = generate_answer(query= query)
            st.divider()
            
            st.subheader("Answer:")
        
            st.write(answer)

    except:
        st.error("⚠️ You must fetch the transcript first.")
            












