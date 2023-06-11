
from typing import Set
from backend.biogpt import biogpt_response
import streamlit as st
from streamlit_chat import message

# Initialize session state variables
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def main():
    st.header("BioGpt ChatBot")
    prompt = st.text_input("Prompt", placeholder="Enter your message here...") or st.button("Submit")

    if prompt:
        with st.spinner("Generating response..."):
            generated_response, chat_history = biogpt_response(query=prompt, chat_history=st.session_state["chat_history"])
            formatted_response = f"{generated_response}"

            # Update session state
            st.session_state["chat_history"] = chat_history
            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)

    if st.session_state["chat_answers_history"]:
        for generated_response, user_query in zip(
            st.session_state["chat_answers_history"],
            st.session_state["user_prompt_history"],
        ):
            message(user_query, is_user=True)
            message(generated_response)

if __name__ == "__main__":
    main()
