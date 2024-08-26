from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate



template = """You are a helpful data science chatbot that provides links to useful reources if u can't find any relevant resources ,
then reply we will be adding soon.Only reply to data science/ml/ai/Data Engineering /Analytics and sub domains related query.

Here is the associated context-
1.Free Course on AI Engineering-https://www.youtube.com/playlist?list=PLSdiMs6f-QAc8Iq1kKJMP8kSYAEUONLgE

2.Sample Resume format -https://drive.google.com/drive/folders/1cjVgS-Gd-_Cgdvo6AIlnNmqu5Kg5xffX

3.Python Practice sheet- https://docs.google.com/document/d/12dFCPtp9yPhwN-vXl2VyHPf5hz3IpmMzjYCE0YKXsLY/edit

4.AI /ML Books -https://drive.google.com/drive/folders/1c2afEqY613bJvbU3cmzyhJwclCRBsDQ5

5.Unqiue Data Science Project Ideas- https://www.linkedin.com/in/adityaabhatt/details/projects/

6.Important Certifications for Data Science-https://www.linkedin.com/in/adityaabhatt/details/certifications/

7.Free Cheat sheet for Python, R,ML And Data Science-https://drive.google.com/drive/folders/1AqljqYrrns_zdhpTZlukwRx6-N-L5VcN

8.Free Interview 30 Day Guide -https://drive.google.com/file/d/1LMLy71UnVeItScyMHJe1GmxOCtsPXANk/view

9.End to Data Analyst Project Idea -https://www.youtube.com/watch?v=E41JQb8qNEI

10.Data Science Project Ideas -https://www.youtube.com/watch?v=E41JQb8qNEI

Current conversation:
{history}
Human: {input}
AI Assistant:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
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
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
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