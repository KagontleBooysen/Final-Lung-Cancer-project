import streamlit as st
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Load the pre-trained BERT model and tokenizer for question answering
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Sample context for the chatbot to answer questions from
context = """
Lung cancer is a type of cancer that begins in the lungs, characterized by uncontrolled cell growth in the lung tissues. It is a significant health challenge in South Africa, with an estimated 8,000 new cases diagnosed annually. Lung cancer is the leading cause of cancer-related deaths among men and one of the top five cancers affecting women. The primary risk factor for lung cancer is smoking, with approximately 20% of the adult population identified as smokers, contributing to its high incidence. Environmental factors, such as exposure to asbestos and industrial pollutants, also play a role. Additionally, South Africa’s high HIV/AIDS prevalence, with about 13% of the adult population living with HIV, exacerbates the lung cancer burden, as immunocompromised individuals are at higher risk. Late-stage diagnosis is common due to limited access to healthcare services and inadequate screening programs, resulting in poorer outcomes. Treatment access is further hindered by the high costs associated with chemotherapy, radiation, and surgical interventions, which are often beyond the reach of many South Africans relying on the overburdened public healthcare system. Public health efforts, including anti-smoking campaigns and initiatives for early detection, are ongoing, but there is a pressing need for more comprehensive and accessible screening programs, along with enhanced support systems for patients and their families.
"""

def main():
    st.set_page_config(page_title="Lung Cancer Chatbot", page_icon="🔍")

    # Custom CSS to style the background and chat container
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url('https://www.utsouthwestern.edu/newsroom/articles/year-2024/assets/lung-cancer-header.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .chat-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 400px;
            max-width: 100%;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            display: none;
        }
        .chat-trigger {
            position: fixed;
            bottom: 20px;
            right: 20px;
            cursor: pointer;
            z-index: 1000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add chat trigger image
    st.markdown(
        """
        <img src="https://cdn3d.iconscout.com/3d/premium/thumb/chat-bot-5379962-4497578.png?f=webp" alt="Chat to us" class="chat-trigger" onclick="toggleChat()" />
        """,
        unsafe_allow_html=True
    )

    # Chat container HTML and JavaScript to toggle visibility
    st.markdown(
        """
        <div class="chat-container" id="chat-container">
            <h2 style='text-align:center;'>Lung Cancer Chatbot</h2>
            <form id="chat-form">
                <input type="text" id="question" placeholder="Ask a question..." style="width: 100%; padding: 10px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px;" required>
                <button type="button" onclick="submitQuestion()" style="width: 100%; padding: 10px; border: none; background-color: #28a745; color: white; border-radius: 4px; cursor: pointer;">Ask</button>
            </form>
            <div class="response" id="response" style="margin-top: 20px; padding: 10px; background-color: #e9ecef; border-radius: 4px; word-break: break-word;"></div>
        </div>
        <script>
        function toggleChat() {
            var chatContainer = document.getElementById('chat-container');
            if (chatContainer.style.display === 'none' || chatContainer.style.display === '') {
                chatContainer.style.display = 'block';
            } else {
                chatContainer.style.display = 'none';
            }
        }

        function submitQuestion() {
            var question = document.getElementById('question').value;
            var responseDiv = document.getElementById('response');
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: question })
            })
            .then(response => response.json())
            .then(data => {
                if (data.answer) {
                    responseDiv.textContent = data.answer;
                } else {
                    responseDiv.textContent = 'Error: ' + data.error;
                }
            });
        }
        </script>
        """,
        unsafe_allow_html=True
    )

    # Streamlit form for handling the question submission
    with st.form(key='chat_form'):
        question = st.text_input('Ask a question:')
        submit_button = st.form_submit_button('Ask')

    if submit_button:
        if question:
            try:
                # Tokenize the input message and context
                inputs = tokenizer.encode_plus(question, context, return_tensors='pt')

                # Get the model's output
                with torch.no_grad():
                    outputs = model(**inputs)

                # Extract the answer start and end logits
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                # Get the most likely start and end token positions
                start_index = torch.argmax(start_logits, dim=-1).item()
                end_index = torch.argmax(end_logits, dim=-1).item()

                # Check if the indices are valid
                if start_index <= end_index and start_index < len(inputs['input_ids'][0]) and end_index < len(inputs['input_ids'][0]):
                    # Convert token indices back to tokens
                    input_ids = inputs['input_ids'].squeeze().tolist()
                    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[start_index:end_index+1])

                    # Clean the answer and convert it to lowercase
                    answer = tokenizer.convert_tokens_to_string(answer_tokens).lower()
                else:
                    answer = "I'm sorry, I don't have the information you are looking for."

                # Display the answer
                st.write(f"**Answer:** {answer}")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning('Please enter a question.')

if __name__ == '__main__':
    main()
