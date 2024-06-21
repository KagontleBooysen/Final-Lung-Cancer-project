import streamlit as st
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Load the pre-trained BERT model and tokenizer for question answering
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Sample context for the chatbot to answer questions from
context = """
Lung cancer, a prevalent issue in South Africa, begins with uncontrolled cell growth in lung tissues, often exacerbated by smoking, which affects approximately 20% of adults. This condition, characterized by its high incidence among men and women, claims thousands of lives annually and ranks among the country's top cancers. Symptoms typically include persistent coughing, chest pain, difficulty breathing, hoarseness, and frequent respiratory infections. Users often inquire about the risk factors, such as smoking, asbestos exposure, and environmental pollutants, which heighten susceptibility. Additionally, with a significant portion of the population living with HIV/AIDS, questions arise about how this immunocompromised status increases the risk of lung cancer. Due to limited healthcare access and late-stage diagnoses, patients frequently ask about available treatments like chemotherapy, radiation, and surgery, alongside the challenges of affordability and access in the public healthcare system. Public health initiatives, including anti-smoking campaigns and early detection programs, aim to mitigate these challenges, yet there remains a critical need for enhanced screening initiatives and comprehensive support systems for patients and families navigating this complex disease landscape.
"""

def get_answer(question):
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

            # Clean the answer and convert it to string
            answer = tokenizer.convert_tokens_to_string(answer_tokens)
        else:
            answer = "I'm sorry, I don't have the information you are looking for."

        return answer

    except Exception as e:
        return f"Error: {e}"

# Function to toggle chat visibility
def toggle_chat():
    st.session_state.show_chat = not st.session_state.show_chat

# Main function
def main():
    st.set_page_config(page_title="Lung Cancer Chatbot", page_icon="🔍")

    if 'show_chat' not in st.session_state:
        st.session_state.show_chat = False

    # Custom CSS to style the background and chat container
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://www.utsouthwestern.edu/newsroom/articles/year-2024/assets/lung-cancer-header.jpg");
            background-size: cover;
        }
        .chat-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 400px;
            max-width: 100%;
            background: rgba(255, 255, 255, 1);
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            display: none;
            z-index: 1000;
        }
        .chat-trigger {
            position: fixed;
            bottom: 20px;
            right: 20px;
            cursor: pointer;
            z-index: 1001;
        }
        .response {
            margin-top: 20px;
            padding: 10px;
            background-color: #ffffff;
            border-radius: 4px;
            word-break: break-word;
            color: #000000; /* Black text color */
        }
        .chat-form {
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #ffffff; /* White background */
            color: #ffffff; /* Black text color */
        }
        .chat-button {
            width: 100%;
            padding: 10px;
            border: none;
            background-color: #28a745; /* Green button */
            color: white; /* White text */
            border-radius: 4px;
            cursor: pointer;
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
    chat_display = "block" if st.session_state.show_chat else "none"
    st.markdown(
        f"""
        <div class="chat-container" id="chat-container" style="display: {chat_display};">
            <h2 style='text-align:center; color: #000000;'>Lung Cancer Chatbot</h2>
            <form id="chat-form">
                <input type="text" id="question" placeholder="Ask a question..." class="chat-form" required>
                <button type="button" onclick="submitQuestion()" class="chat-button">Ask</button>
            </form>
            <div class="response" id="response"></div>
        </div>
        <script>
        function toggleChat() {{
            var chatContainer = document.getElementById('chat-container');
            if (chatContainer.style.display === 'none' || chatContainer.style.display === '') {{
                chatContainer.style.display = 'block';
            }} else {{
                chatContainer.style.display = 'none';
            }}
            fetch('/toggle_chat');
        }}

        function submitQuestion() {{
            var question = document.getElementById('question').value;
            var responseDiv = document.getElementById('response');
            if (question) {{
                fetch('/', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{ 'question': question }})
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.answer) {{
                        responseDiv.textContent = data.answer;
                    }} else {{
                        responseDiv.textContent = 'Error: ' + data.error;
                    }}
                }});
            }}
        }}
        </script>
        """,
        unsafe_allow_html=True
    )

    # Streamlit form for handling the question submission
    with st.form(key='chat_form'):
        st.title(':blue[Ask a question:]')
        question = st.text_input('')
        submit_button = st.form_submit_button('Ask')

    if submit_button and question:
        answer = get_answer(question)
        st.write(f"<span style='color: black; font-size: 30px;'>**Answer:** {answer}</span>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
