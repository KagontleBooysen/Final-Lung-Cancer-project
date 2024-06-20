from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained pipeline for conversational AI
chatbot = pipeline("conversational")

# Static response about lung cancer in South Africa
lung_cancer_info = """
Lung cancer is a type of cancer that begins in the lungs, characterized by uncontrolled cell growth in the lung tissues. It is a significant health challenge in South Africa, with an estimated 8,000 new cases diagnosed annually. Lung cancer is the leading cause of cancer-related deaths among men and one of the top five cancers affecting women. The primary risk factor for lung cancer is smoking, with approximately 20% of the adult population identified as smokers, contributing to its high incidence. Environmental factors, such as exposure to asbestos and industrial pollutants, also play a role. Additionally, South Africa’s high HIV/AIDS prevalence, with about 13% of the adult population living with HIV, exacerbates the lung cancer burden, as immunocompromised individuals are at higher risk. Late-stage diagnosis is common due to limited access to healthcare services and inadequate screening programs, resulting in poorer outcomes. Treatment access is further hindered by the high costs associated with chemotherapy, radiation, and surgical interventions, which are often beyond the reach of many South Africans relying on the overburdened public healthcare system. Public health efforts, including anti-smoking campaigns and initiatives for early detection, are ongoing, but there is a pressing need for more comprehensive and accessible screening programs, along with enhanced support systems for patients and their families.
"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the message from the POST request
        data = request.get_json()
        user_input = data['message']
        
        # Check if the user is asking about lung cancer in South Africa
        if 'South Africa' in user_input or 'lung cancer' in user_input:
            # Return the predefined information about lung cancer in South Africa
            return jsonify({'response': lung_cancer_info})
        
        # Let the chatbot generate a response for other queries
        bot_response = chatbot(user_input)[0]['generated_text']

        # Return the answer as a JSON response
        return jsonify({'response': bot_response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
