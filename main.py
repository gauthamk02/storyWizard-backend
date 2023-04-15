from flask import Flask, jsonify, request, json
import os
import openai
import pandas as pd
import base64
import pickle
import os
import requests

openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

df = pd.read_csv('data.csv')

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
        raise Exception("Non-200 response for image generation: " + str(response.text))

    data = response.json()

    for i, image in enumerate(data["artifacts"]):

            return image["base64"]

def save_story(title: str, story: str, img: str):
    
    #save img to images folder
    img_file = f"./images/{title}.png"
    with open(img_file, "wb") as f:
        f.write(base64.b64decode(img))

    global df

    temp_df = pd.DataFrame({
        "id": [len(df)+1],
        "title": [title],
        "story": [story],
        "img": [img_file]
    })       
    
    df = pd.concat([df, temp_df], ignore_index=True)
    df.to_csv('data.csv', index=False)

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Hello World!'})

@app.route('/generate', methods=['GET'])
def generate():
    topic = json.loads(request.data)['topic']
    title, story = generate_story(topic)
    print(f"Title: {title}")
    print(f"Story: {story}")
    prompts = generate_prompts(story)
    print(f"Prompts: {prompts}")
    img = generate_image(prompts)
    print("Image generated")
    save_story(title, story, img)

    return jsonify({'title': title, 'story': story, 'img': img})

@app.route('/get_n_stories', methods=['GET'])
def get_n_stories():
    n = json.loads(request.data)['n']
    stories = df.sample(n=n).to_dict('records')
    return jsonify({'stories': stories})

if __name__ == '__main__':
    app.run(debug=True)