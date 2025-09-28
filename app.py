import streamlit as st
from rag_utils import (
    get_yt_transcript,
    generate_answer, 
    load_api_key
)

st.set_page_config(page_title="Youtube Q&A", page_icon="🎥", layout="centered")

st.title("🎥 Youtube Transcript Q&A")
st.divider()


url = st.sidebar.text_input("Youtube Video URL")

palceholder = st.empty()

process_url_button = st.sidebar.button("Process URL")

if process_url_button:
    if not url:
        st.sidebar.error("You must provide the URL first")
    else:
        api_key = load_api_key()
        for status in get_yt_transcript(url= url, api_key= api_key):
            palceholder.status(status)

query = palceholder.text_area("Question")

query_button = st.button("Enter")

if query_button:
    if not query:
        st.error("No query found")
    
    else:
        try:
            with st.spinner("Thinking... for better response."):
                answer = generate_answer(query= query)
                st.divider()

                st.subheader("Answer:")

                st.write(answer)
        except Exception as e:
            st.error(f"{str(e)}")








