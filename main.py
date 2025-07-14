# main.py
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, DocumentCompressorPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers.long_context_reorder import LongContextReorder
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
import os
from example import fewshot_examples
from dotenv import load_dotenv
load_dotenv()

os.makedirs('sqlite_cache', exist_ok=True)
set_llm_cache(SQLiteCache(database_path = './sqlite_cache/sqlite_cache.db'))

def load_vector_store():
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')
    vector_store = Chroma(
        persist_directory='./farm',
        embedding_function=embeddings,
        collection_name='farm'
    )
    return vector_store

def create_chain(vector_store):
    chat_llm = ChatOpenAI(model = 'gpt-4.1', streaming = True)
    helper_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    output_parser = StrOutputParser()
    
    tuple_fewshow_examples = []
    for example in fewshot_examples:
        tuple_fewshow_examples.append(('human', example['question']))
        tuple_fewshow_examples.append(('ai', example['answer']))


    prompt = ChatPromptTemplate.from_messages([
        ('system', '당신은 농업분야의 전문가 입니다. 사용자의 질문에 아래 참고 문서를 바탕으로 답변해주세요.'),
        *tuple_fewshow_examples,
        MessagesPlaceholder(variable_name='history'),
        ('human', "질문:{question} \n\n 참고 문서\n{context}")
    ])
    
    child_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150,
    separators = ['\n\n', '\n', '.', ' ', '']
    )
    
    docstore_path = './parent_docstore'
    store = create_kv_docstore(LocalFileStore(docstore_path))
        
    parent_retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    child_splitter=child_splitter
    )
            
    reorder = LongContextReorder()
    extractor = LLMChainExtractor.from_llm(helper_llm)

    compressor = DocumentCompressorPipeline(
        transformers=[reorder, extractor]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=parent_retriever
    )

    chain = prompt | chat_llm | output_parser
    return chain, compression_retriever

def get_answer(chain, retriever, query, history):
    context_docs = retriever.invoke(query)
    llm_answer = chain.invoke({
        "question" : query,
        "context" : '\n\n'.join([doc.page_content for doc in context_docs]),
        "history" :  history
    })
    return llm_answer

def get_answer_stream(chain, retriever, query, history):
    context_docs = retriever.invoke(query)
    llm_answer = chain.stream({
        "question" : query,
        "context" : '\n\n'.join([doc.page_content for doc in context_docs]),
        "history" :  history
    })
    return llm_answer