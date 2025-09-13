from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template='''
    You are a helpful assistant.
    Answer the question based on the context provided.
    If the information is insufficient simply say I don't know.

    {context}\n
    Question: {question}
''',
input_variables=['context','question']
)

template.save('template.json')