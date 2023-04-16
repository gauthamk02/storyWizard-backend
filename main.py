from flask import Flask, jsonify, request, json, send_from_directory
import os
import openai
import pandas as pd
import base64
import os
import requests
import numpy as np
import google.cloud.texttospeech as tts

from flask_cors import CORS

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service.json'
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

stories_file = 'data/stories.csv'
session_file = 'data/session.csv'

if not os.path.exists(stories_file):
    df = pd.DataFrame({
        "id": [],
        "title": [],
        "story": [],
        "img": []
    })
    df.to_csv(stories_file, index=False)

if not os.path.exists(session_file):
    df = pd.DataFrame({
        "id": [],
        "sess_id": [],
        "story_id": [],
        "role": [],
        "content": []
    })
    df.to_csv(session_file, index=False)

stories_df = pd.read_csv(stories_file)
session_df = pd.read_csv(session_file)

def generate_story(topic: str) -> str:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Generate a 4 paragraph children's story with title about {topic} that contains a moral."}
        ]
    )
    content = completion.choices[0].message.content
    content = content.encode().decode('unicode_escape')
    title = content.split('\n')[0]
    title = title.replace('Title: ', '')
    story = content[content.find('\n'):]
    story = story.lstrip()

    return title, story

def generate_prompts(story: str):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Create one text to image prompts that will be suitable as the title image of the below given story. Do not include the character names, instead include only the characters physical description.\n\n{story}"}
        ]
    )
    content = completion.choices[0].message.content
    content = content.encode().decode('unicode_escape')
    if ':' in content:
        content = content[content.find(':')+1:]
    content = content.strip()
    return content

def generate_image(prompt: str):
    engine_id = "stable-diffusion-512-v2-1"
    api_host = os.getenv('API_HOST', 'https://api.stability.ai')
    api_key = os.getenv("STABILITYAI_API_KEY")

    if api_key is None:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": f"{prompt}"
                }
            ],
            "cfg_scale": 7,
            "clip_guidance_preset": "FAST_BLUE",
            "height": 512,
            "width": 512,
            "samples": 1,
            "steps": 30,
        },
    )

    if response.status_code != 200:
        raise Exception(
            "Non-200 response for image generation: " + str(response.text))

    data = response.json()

    for i, image in enumerate(data["artifacts"]):

        return image["base64"]

def save_story(title: str, story: str, img: str, img_filename: str, audio_filename: str):

    with open(img_filename, "wb") as f:
        f.write(base64.b64decode(img))

    global stories_df

    temp_df = pd.DataFrame({
        "id": [len(stories_df)+1],
        "title": [title],
        "story": [story],
        "img": [request.root_url + 'images/' + title + '.png'],
        "audio": [request.root_url + 'audios/' + title + '.wav']
    })

    stories_df = pd.concat([stories_df, temp_df], ignore_index=True)
    stories_df.to_csv(stories_file, index=False)

def get_followup_response(session_id: int, story_id: int, question: str):
    global session_df

    story = stories_df[stories_df['id'] == story_id]['story'].values[0]
    system_msg = f"You are an assistant that answers the questions to the children's "\
                 "story given below. You should answer the questions descriptively in a "\
                 "way that a child can understand them. If the question asked is unrelated "\
                 "to the story, do not answer the question and instead reply by asking the "\
                 "user to ask questions related to the story."\
                 "\n\n"\
                 f"Story: {story}"

    temp_df = pd.DataFrame({
        "id": [len(session_df)+1],
        "sess_id": [session_id],
        "story_id": [story_id],
        "role": ["user"],
        "content": [question]
    })

    session_df = pd.concat([session_df, temp_df], ignore_index=True)

    messages = session_df[session_df['sess_id']
                          == session_id][["id", "role", "content"]]
    messages = messages.sort_values(by=['id'])
    messages = messages[['role', 'content']]
    messages = messages.to_dict('records')

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            *messages
        ]
    )

    content = completion.choices[0].message.content
    content = content.encode().decode('unicode_escape')

    temp_df = pd.DataFrame({
        "id": [len(session_df)+1],
        "sess_id": [session_id],
        "story_id": [story_id],
        "role": ["assistant"],
        "content": [content]
    })

    session_df = pd.concat([session_df, temp_df], ignore_index=True)
    session_df.to_csv(session_file, index=False)

    return content

def text_to_wav(text: str, title, dest, voice_name = "en-IN-Wavenet-A"):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16,
                                   speaking_rate=0.8)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    filename = f"{dest + '/' + title}.wav"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated speech saved to "{filename}"')

    return filename

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Hello World!'})

@app.route('/images/<path:path>', methods=['GET'])
def get_image(path):
    return send_from_directory('images', path)

@app.route('/audios/<path:path>', methods=['GET'])
def get_audio(path):
    return send_from_directory('audios', path)

@app.route('/generate', methods=['GET'])
def generate():
    topic = request.args.get('topic')
    title, story = generate_story(topic)
    print(f"Title: {title}")
    print(f"Story: {story}")
    prompts = generate_prompts(story)
    print(f"Prompts: {prompts}")
    img = generate_image(prompts)
    print("Image generated")
    audio_file = text_to_wav(story, title, "./audios")
    print("Audio generated")
    img_filename = f"./images/{title}.png"
    save_story(title, story, img, img_filename, audio_file)

    return jsonify({'title': title, 'story': story, "id": len(stories_df),
                     'img': request.root_url + 'images/' + title + '.png', 'audio': request.root_url + 'audios/' + title + '.wav'})

@app.route('/get_story', methods=['GET'])
def get_story():
    story_id = int(request.args.get('id'))
    story = stories_df[stories_df['id'] == story_id].to_dict('records')
    return jsonify({'story': story})

@app.route('/get_n_stories', methods=['GET'])
def get_n_stories():
    n = int(request.args.get('n'))
    stories = stories_df.sample(n=n).to_dict('records')
    return jsonify({'stories': stories})

@app.route('/get_story_count', methods=['GET'])
def get_story_count():
    return jsonify({'count': len(stories_df)})

@app.route('/get_followup', methods=['GET'])
def get_followup():
    session_id = int(request.args.get('session_id'))
    story_id = int(request.args.get('story_id'))
    question = request.args.get('question')
    response = get_followup_response(session_id, story_id, question)
    audio_file = text_to_wav(response, f"temp", "./audios")
    return jsonify({'response': response, 'audio': request.root_url + 'audios/' + 'temp' + '.wav'})

def transcribe_file(audio):
    """Transcribe the given audio file."""
    from google.cloud import speech

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(content=audio)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print("Transcript: {}".format(result.alternatives[0].transcript))
        return result.alternatives[0].transcript

@app.route('/post_followup_audio', methods=['POST'])
def get_text():
    # get the audio data from form
    audio_file = request.files['audio']
    sess_id = request.form['session_id']
    story_id = request.form['story_id']
    text = transcribe_file(audio_file.read())
    if text is None:
        return jsonify({'response': 'Sorry, I could not understand you. Please try again.'})
    response = get_followup_response(sess_id, story_id, text)
    return jsonify({'response': response})

if __name__ == '__main__':

    app.run(debug=True)