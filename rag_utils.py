import os
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders.youtube import YoutubeLoader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv


#---------CONFIGURATION CONSTANTS---------#

LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.6
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "youtube-transcript"
VECTOR_DB_PATH = Path(__file__).parent/"resources/vector_store"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80



#------INITIALIZE OBJECTS---------#
llm = None
vector_store = None


#---------PURE FUNCTIONS----------#
def load_api_key():
    """Load and read the API key from streamlit or .env file."""
    load_dotenv()

    #for cloud deployment
    try:
        import streamlit as st
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except ImportError:
        pass 
    except Exception:
        pass

    #for local development
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Ensure .env file exists with OPENAI_API_KEY or Streamlt secrets are configured"
        )
    
    return api_key



def initialize_components(api_key: str):
    """Initialize LLM and vector store once."""

    global llm, vector_store

    if llm is None:
        llm = ChatOpenAI(
            model = LLM_MODEL,
            temperature= TEMPERATURE,
            api_key= api_key
        )

    if vector_store is None:

        embedding_model = OpenAIEmbeddings(
            model = EMBEDDING_MODEL,
            api_key= api_key
        )

        vector_store = Chroma(
            collection_name= COLLECTION_NAME,
            embedding_function= embedding_model,
            persist_directory= str(VECTOR_DB_PATH)
        )



def get_video_id(url: str) -> str:
    """Extract Youtube video ID from the given URL"""

    return YoutubeLoader.extract_video_id(url)


def get_yt_transcript(url:str, api_key: str):
    """Fetch the Youtube transcript and store it in the vector store"""

    yield "Initialize components..."
    initialize_components(api_key= api_key)

    yield "Resetting vector store.." #clear any existing documents
    vector_store.reset_collection()

    yield "Fetching Video ID..."
    video_id = get_video_id(url= url)

    yield "Loading transcript..."
    try:
        yt_api = YouTubeTranscriptApi()
        transcript_list = yt_api.list(video_id= video_id)
        transcript_en = transcript_list.find_generated_transcript(language_codes=['en'])
        raw_transcript = transcript_en.fetch()
        # raw_transcript looks like - [{"text": "some words", "start": 12.3, "duration": 4.2}, ...]

        #Join all the transcript texts into one long string.
        transcript = " ".join(transcript.text for transcript in raw_transcript)

    except TranscriptsDisabled:
        raise RuntimeError(f"No transcript availabe for the video id: {video_id}")
    except Exception as e:
        raise RuntimeError(f"Error fetching transcript: {str(e)}")


    yield "Splitting transcript into chunks..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )

    chunks = splitter.create_documents([transcript])

    yield "Adding chunks to vector store..."
    vector_store.add_documents(chunks)

    yield "Finished adding documents to vector database"


def generate_answer(query:str) ->str:
    """retrieved context from the vector DB and anwers the query"""

    if vector_store is None:
        raise RuntimeError("VectorDB is not initialized. Please process a Youtube URL first")
    
    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {'k':3})

    #combines muliple retrieved chunks into one context string
    def format_doc(retrieved_doc):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_doc)
        
        return context_text
    
    prompt = load_prompt('template.json')

    parser = StrOutputParser()


    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_doc),
        "question": RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | llm | parser

    #pass query as string directly
    return final_chain.invoke(query)



if __name__ == "__main__":
    
    api_key = load_api_key()
    url ="https://www.youtube.com/watch?v=aircAruvnKk&t=4s"

    get_yt_transcript(url = url, api_key= api_key)
    answer = generate_answer("Summarize the key concepts of neural Network")
    print(f"Answer:{answer}")

