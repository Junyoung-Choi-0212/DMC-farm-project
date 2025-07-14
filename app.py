from example import fewshot_examples
from main import load_vector_store, create_chain, get_answer_stream
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import uuid

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = load_vector_store()

if 'chain' not in st.session_state:
    st.session_state.chain,st.session_state.retriever  = create_chain(
        st.session_state.vector_store)

if 'all_memory' not in st.session_state:
    st.session_state.all_memory = {}

if session_id not in st.session_state.all_memory:
    st.session_state.all_memory[session_id] = ConversationBufferMemory(return_messages = True)

user_memory = st.session_state.all_memory[session_id]
st.title('농업 종사자를 위한 챗봇')

for message in user_memory.chat_memory.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message('user'):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message('assistant'):
            st.write(message.content)
    
if user_input := st.chat_input('채팅을 입력하세요.'):
    with st.chat_message('user'):
        st.write(user_input)
    user_memory.chat_memory.add_user_message(user_input)
    
    history = user_memory.load_memory_variables({})['history']
    with st.chat_message('assistant'):
        collected_chunks = []                        # ① 토큰을 모을 리스트

        def _stream_and_collect():
            for chunk in get_answer_stream(
                st.session_state.chain,
                st.session_state.retriever,
                user_input,
                history,
            ):
                collected_chunks.append(chunk)       # ② 리스트에 누적
                yield chunk                          # ③ 동시에 화면으로 흘려보냄

        st.write_stream(_stream_and_collect())       # ④ 실시간 출력

    assistant_text = "".join(collected_chunks)       # ⑤ 토큰 합치기

    user_memory.chat_memory.add_ai_message(assistant_text)