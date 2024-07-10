import streamlit as st
import testing

st.title("Weather and Clothing Suggestions")

user_input = st.text_input("Enter your question:")

if st.button("Get Response"):
    if user_input:
        response = testing.final_ans(user_input)
        st.write(response["output"])
    else:
        st.write("Please enter a question.")