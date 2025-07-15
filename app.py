from chat_history_db import create_chat_history_table, insert_chat_history, get_all_session_ids_with_time, load_chat_history
from example import fewshot_examples
from main import load_vector_store, create_chain, get_answer_stream
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import uuid

# 🟢 테이블 생성
create_chat_history_table()

# ✅ 새 채팅 버튼
if st.sidebar.button("🆕 새 채팅 시작"):
    new_session_id = str(uuid.uuid4())
    st.session_state.session_id = new_session_id
    st.session_state.new_chat = True
    st.session_state.selected_session_index = 0
    st.session_state.loaded_from_db = {}
    st.rerun()

st.sidebar.markdown("---")

# ✅ 세션 목록 표시
st.sidebar.header("💬 이전 대화 불러오기")
display_names, session_ids = get_all_session_ids_with_time()
st.sidebar.write("🗂 세션 개수:", len(session_ids))
st.sidebar.write("세션 목록:", display_names)

if "selected_session_index" not in st.session_state:
    st.session_state.selected_session_index = 0

# ✅ 세션 선택 박스
if session_ids:
    selected_index = st.sidebar.selectbox(
        "세션 선택",
        range(len(session_ids)),
        format_func=lambda i: display_names[i],
        index=st.session_state.selected_session_index,
        key="session_selectbox"
    )

    # ✅ 선택된 세션 ID
    session_id_to_select = session_ids[selected_index]

    # ✅ 세션 변경 시 처리
    if "new_chat" in st.session_state or st.session_state.selected_session_index != selected_index:
        st.session_state.session_id = session_id_to_select
        st.session_state.selected_session_index = selected_index

        # ✅ 캐시 제거 (단, rerun 전에만!)
        if "loaded_from_db" in st.session_state:
            st.session_state.loaded_from_db.pop(session_id_to_select, None)

        if "new_chat" in st.session_state:
            del st.session_state["new_chat"]

        st.rerun()

    # ✅ 세션 확정
    session_id = st.session_state.session_id
else:
    st.sidebar.write("저장된 세션이 없습니다.")
    st.stop()

# ✅ 벡터 DB & 체인 구성
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = load_vector_store()

if 'chain' not in st.session_state:
    st.session_state.chain, st.session_state.retriever = create_chain(
        st.session_state.vector_store
    )

if "all_memory" not in st.session_state:
    st.session_state.all_memory = {}

if "loaded_from_db" not in st.session_state:
    st.session_state.loaded_from_db = {}

# ✅ 세션별 memory 생성 or 히스토리 주입
if session_id not in st.session_state.loaded_from_db:
    chat_records = load_chat_history(session_id)

    messages = []
    for question, answer in chat_records:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))

    memory = ConversationBufferMemory(return_messages=True)
    memory.chat_memory.messages = messages
    st.session_state.all_memory[session_id] = memory

    st.session_state.loaded_from_db[session_id] = True
    st.rerun()
    st.stop()

# ✅ session_id에 해당하는 memory를 항상 안전하게 보장
if session_id not in st.session_state.all_memory:
    st.session_state.all_memory[session_id] = ConversationBufferMemory(return_messages=True)

user_memory = st.session_state.all_memory[session_id]

# ✅ 채팅 UI 출력
st.title('농업 종사자를 위한 챗봇')

for message in user_memory.chat_memory.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message('user'):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message('assistant'):
            st.write(message.content)

# ✅ 입력 처리
if user_input := st.chat_input('채팅을 입력하세요.'):
    if "new_chat" in st.session_state:
        del st.session_state["new_chat"]

    with st.chat_message('user'):
        st.write(user_input)
    user_memory.chat_memory.add_user_message(user_input)

    history = user_memory.load_memory_variables({})['history']

    with st.chat_message('assistant'):
        collected_chunks = []

        def _stream_and_collect():
            for chunk in get_answer_stream(
                st.session_state.chain,
                st.session_state.retriever,
                user_input,
                history,
            ):
                collected_chunks.append(chunk)
                yield chunk

        st.write_stream(_stream_and_collect())

    assistant_text = "".join(collected_chunks)
    user_memory.chat_memory.add_ai_message(assistant_text)

    insert_chat_history(session_id, user_input, assistant_text)