from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders.youtube import YoutubeLoader
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------- CONSTANTS ----------
TEMPERATURE = 0.8
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
NAME_OF_DB = "Youtube_transcript"
VECTORDB_PATH = Path(__file__).parent / "resources/vector_store"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

#Initialize objects
llm = None
vector_store = None


def initialize_components():
    """Initialize LLM and vector store once."""

    global llm, vector_store

    if llm is None:

        llm = ChatOpenAI(
            model= LLM_MODEL,
            temperature= TEMPERATURE
        )

    if vector_store is None:

        embedding_model = OpenAIEmbeddings(
            model= EMBEDDING_MODEL
        )

        vector_store = Chroma(
            collection_name = NAME_OF_DB,
            embedding_function = embedding_model,
            persist_directory = str(VECTORDB_PATH)
        )


def get_video_id(url: str) -> str:
    """Extract YouTube video ID from the URL."""

    return YoutubeLoader.extract_video_id(url)


def get_yt_transcript(url: str):
    """
    Fetch the YouTube transcript and store it in the vector store.
    """

    yield "Initializing components..."
    initialize_components()

    yield "Resetting vector store..."
    vector_store.reset_collection()  #Clears any existing documents

    yield "Fetching Video ID..."
    video_id = get_video_id(url)

    yield "Loading transcript..."
    try:
        yt_api = YouTubeTranscriptApi()
        transcript_list = yt_api.list(video_id= video_id)
        transcript_en = transcript_list.find_generated_transcript(language_codes=['en'])
        raw_transcript = transcript_en.fetch()
        #raw _transcript looks like -{"text": "some words", "start": 12.3, "duration": 4.2}

        #Joins all the caption texts into one long string.
        transcript = " ".join(transcript.text for transcript in raw_transcript)
        
    except TranscriptsDisabled:
        raise RuntimeError(f"No transcript available for the video id: {video_id}")


    yield "Splitting transcript into chunks..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.create_documents([transcript])

    yield "Adding chunks to vector store..."
    vector_store.add_documents(chunks)

    yield "Finished adding documents to vectore database"

def generate_answer(query: str) -> str:
    """Retrieve context from the vector DB and answer the query."""

    if vector_store is None:
        raise RuntimeError("Vector DB is not initialized")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Combines multiple retrieved chunks into one context string
    def format_doc(retrieved_doc):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_doc)

        return context_text

    prompt = load_prompt("template.json")

    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_doc),
        "question": RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | llm | parser

    # Pass query as dict
    return final_chain.invoke(query)



if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=fNk_zzaMoSs&t=227s"
    get_yt_transcript(url=url)
    answer = generate_answer("Summarise the key concepts explained in the video.")
    print("\n--- Answer ---\n", answer)

    
    