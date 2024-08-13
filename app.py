import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Load environment variables from a .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
key = os.getenv('OPEN_AI_KEY')

# Initialize the Flask application
app = Flask(__name__)

# Define a route to classify email content, accessible via POST requests
@app.route('/classify', methods=['POST'])
def classify_email():
    try:
        # Get JSON data from the request
        data = request.get_json()
        # Retrieve the 'all_email_content' field from the JSON data and strip whitespace
        all_email_content = data.get('all_email_content', '').strip()

        # Check if 'all_email_content' is empty and return an error if it is
        if not all_email_content:
            return jsonify({"error": "Please provide the email content"}), 400

        # Define the prompt template to classify the email content into specific categories
        template = """ 
        Classify the following email content into one of these categories: Acknowledged, Agreed for Meeting, Need Proposal, No Response, Not Connected, Closed Lost, Asked to Connect Later, Need Demo, Unsubscribed.If the email content does not make any sense or is unclear, return "email is senseless."

        Email:
        {all_email_content}

        Classification:
        """
        # Create a prompt template using the defined template
        prompt = PromptTemplate.from_template(template)

        # Initialize the OpenAI LLM (Large Language Model) with the API key and model name
        llm = OpenAI(openai_api_key=key, model_name='gpt-3.5-turbo-instruct')

        # Create a chain where the prompt is passed to the LLM for processing
        llm_chain = prompt | llm

        # Invoke the chain with the email content to get the classification
        result = llm_chain.invoke({"all_email_content": all_email_content})
        # Strip any extra whitespace from the result
        cleaned_result = result.strip()

        # Check if the result is empty and return a message if no classification is available
        if not cleaned_result:
            return jsonify({"classification": "No classification available"}), 200

        # Return the classification result as a JSON response
        return jsonify({"classification": cleaned_result}), 200

    except Exception as e:
        # Handle any exceptions and return an error message with a 500 status code
        return jsonify({"error": str(e)}), 500

# Run the Flask application on all available IP addresses (host '0.0.0.0') on port 8000
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)
