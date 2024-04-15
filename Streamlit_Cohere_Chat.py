import cohere
import streamlit as st
import os

co = cohere.Client(api_key = os.environ.get("COHERE_API_KEY"))
st.title('Chat with Cohere')

if 'messages' not in st.session_state:
    st.session_state.messages = []

chat_log = []

if_connectors_open = {"id": "web-search"}
connectors = []

with st.sidebar:
    preamble: str = st.text_area('System Prompt', value = 'You are a helpful assistant.')
    temperature: float = st.slider('Temperature', 0.0, 1.0, 0.7, step = 0.1)
    max_tokens: int = st.slider('Max Tokens', 1, 4000, 4000, step = 1)
    k: int = st.slider('Top_K', 0, 500, 0, step = 1, help = '确保在每一步生成时仅考虑前k个最有可能的令牌。')
    p: float = st.slider('Top_P', 0.01, 0.99, 0.75, step = 0.01, help = '确保在每一步生成时仅考虑概率之和为p的最有可能的令牌。如果同时启用Top_K和Top_P，则Top_P在Top_K之后起作用。')
    frequency_penalty: float = st.slider('Frequency Penalty', 0.0, 1.0, 0.0, step = 0.1, help = '用于减少生成令牌的重复性。该值越高，对已在提示或先前生成中出现过的令牌施加的惩罚越强，且惩罚力度与其已出现次数成正比。')
    presence_penalty: float = st.slider('Presence Penalty', 0.0, 1.0, 0.0, step = 0.1, help = '用于减少生成令牌的重复性。类似于 Frequency Penalty。但此惩罚会同等应用于所有已出现过的令牌，而不考虑其确切出现频率。')
    connectors_choice = st.toggle('Connectors')
    if connectors_choice:
        connectors.append(if_connectors_open)
        prompt_truncation = 'AUTO'
    else:
        connectors = []
        prompt_truncation = 'OFF'
    if st.button('New Chat'):
        st.session_state.messages = []
        chat_log = []
        st.experimental_rerun()
    if st.button('Check'):
        if st.session_state.messages or chat_log is not None:
            st.write(st.session_state.messages)

user_msg = st.chat_input('Say something...')

if user_msg:
    st.session_state.messages.append(
        {'role': 'user', 'content': user_msg}
    )
    
    response = co.chat(
        preamble = preamble,
        model = 'command-r-plus',
        connectors = connectors,
        prompt_truncation = prompt_truncation,
        chat_history = chat_log,
        message = user_msg,
        temperature = temperature,
        max_tokens = max_tokens,
        k = k,
        p = p,
        frequency_penalty = frequency_penalty,
        presence_penalty = presence_penalty
    )

    st.session_state.messages.append(
        {'role': 'assistant', 'content': response.text}
    )

    chat_log.append(
        {'role': 'USER', 'message': user_msg}
    )
    chat_log.append(
        {'role': 'CHATBOT', 'message': response.text}
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'], unsafe_allow_html = True)

