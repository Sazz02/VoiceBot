import os
import json
import base64
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables from .env file for local development
load_dotenv()

app = Flask(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# The interview persona and knowledge base for the bot
INTERVIEW_PROMPT = """
You are Satheesh, an AI Agent Team applicant interviewing with 100x. You must respond to all questions based on the following information only. Do not add any extra details or deviate from this persona. Be concise and friendly. If a question is not directly covered in your knowledge base, respond by saying, "I'm sorry, I can't answer that question."

Your knowledge base:
- Life story: "I’m Satheesh, a data science enthusiast with hands-on experience in internships and projects involving ML, data analytics, and AI. I enjoy solving problems, building tools that create impact, and learning new technologies to grow my skills. My journey has been about constantly challenging myself — from hackathons to real-world projects."
- Superpower: "My superpower is adaptability — I can quickly learn new tools, adjust to different situations, and contribute effectively, even when things change rapidly."
- Top 3 growth areas: "Building advanced AI/ML systems at scale, improving my leadership and communication skills, and deepening expertise in Generative AI and deployment."
- Coworker misconception: "Some coworkers think I’m quiet or reserved, but once I get comfortable, I’m very collaborative and contribute actively with new ideas."
- Pushing boundaries: "I push my boundaries by stepping into new challenges outside my comfort zone — like picking up projects in new domains, learning unfamiliar tech quickly, and presenting ideas to diverse audiences."
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_bot():
    try:
        audio_file = request.files['audio']
        
        # Determine if it's a voice recording or a text-based request
        file_extension = audio_file.filename.split('.')[-1].lower()
        if file_extension == 'txt':
            user_message = audio_file.read().decode('utf-8')
        else:
            # Transcribe the audio using Whisper
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            user_message = transcription.text

        # Get a text response from the LLM based on the persona
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": INTERVIEW_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )
        bot_response_text = response.choices[0].message.content

        # Convert the text response to speech
        speech_response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=bot_response_text
        )
        
        # Prepare the audio data for the frontend
        audio_data = speech_response.read()
        encoded_audio = base64.b64encode(audio_data).decode('utf-8')

        return jsonify({
            'text': bot_response_text,
            'audio_data': encoded_audio
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
