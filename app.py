from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize ChatAnthropic LLM
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2
)

# Initialize conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=True
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    # Get response from LangChain conversation
    response = conversation.predict(input=user_message)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)