import json

import streamlit as st
from seed_data import seed_from_json, seed_from_pdf
from agent import get_retriever, get_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import requests

BACKEND_URL = 'http://localhost:8500'


def setup_page():
    st.set_page_config(
        page_title='Elly! Your AI Assistant!',
        page_icon="💬",
        layout='wide'
    )


def ini_app():
    setup_page()


def handle_url_pdf():
    collection_name = st.text_input(
        'Your collection in Milvus',
        'data_live',
        help='Enter your collection name in Milvus'
    )

    pdf_path = st.text_input('Enter pdf path: ')

    if st.button("Crawl file"):
        if not collection_name:
            st.error("Crawl Failed! No Collection name found!")
            return
        with st.spinner('Crawling data'):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/seed_url_pdf",
                    data={
                        'file': pdf_path,
                        'collection_name': collection_name
                    }
                )
                if response.status_code == 200:
                    st.success('Seeded PDF successfully!')
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Crawl failed {str(e)}!")


def handle_local_pdf():
    collection_name = st.text_input(
        'Your collection in Milvus',
        'data_live',
        help='Enter your collection name in Milvus'
    )

    pdf_path = st.file_uploader('Enter pdf path: ')

    if st.button("Crawl file"):
        if not collection_name:
            st.error("Crawl Failed! No Collection name found!")
            return
        with st.spinner('Crawling data'):
            try:
                files = {"file": pdf_path}
                response = requests.post(
                    f"{BACKEND_URL}/seed_upload_pdf",
                    files=files,
                    data={'collection_name': collection_name}
                )
                if response.status_code == 200:
                    st.success('Seeded PDF successfully!')
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Crawl failed {str(e)}!")


def setup_sidebar():
    with st.sidebar:
        st.title('Setting!')
        st.header('Source:')
        data_source = st.radio(
            "Choose file type:",
            ['Local', 'URL']
        )
        if data_source == 'Local':
            handle_local_pdf()
        else:
            handle_url_pdf()
        st.header("Find collection to query")
        collection_to_query = st.text_input(
            "Enter collection name to query: "
            "data_test_live",
            help='Enter collection name which you want to query'
        )
        return collection_to_query


def setup_chat_interface():
    st.title('💬 Elly!')
    st.caption('AI assistant!')

    msgs = StreamlitChatMessageHistory(key='langchain_message')

    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            'role': 'assistant',
            'content': 'How can i help you?'
        }]
        msgs.add_ai_message('How can i help you?')

    for msg in st.session_state.messages:
        role = 'assistant' if msg['role'] == 'assistant' else 'human'
        st.chat_message(role).write(msg['content'])

    return msgs


def handle_input(msgs, agent_executor, collection_name):
    if prompt := st.chat_input("Ask me something about tour document!"):
        st.session_state.messages.append({
            'role': 'human',
            'content': prompt
        })
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())

            chat_history = [
                {
                    'role': msg['role'],
                    'content': msg['content']
                }
                for msg in st.session_state.messages[:-1]
            ]
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    data={
                        'prompt': prompt,
                        'collection_name': collection_name,
                        'history': json.dumps(chat_history)
                    }
                )
                if response.status_code == 200:
                    output = response.json()['answer']
                    st.session_state.messages.append({'role': 'assistant', 'content': output})
                    msgs.add_ai_message(output)
                    st.write(output)
                else:
                    st.error(f"ERROR: {response.json().get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")


def main():
    ini_app()
    collection_to_query = setup_sidebar()
    msgs = setup_chat_interface()
    retriever = get_retriever(collection_to_query)
    agent_executor = get_agent(retriever)
    handle_input(msgs, agent_executor, collection_to_query)


if __name__ == "__main__":
    main()
