# Story Wizard - Frontend

Welcome to our interactive story teller project! We created this app as part of a hackathon with one goal in mind: to make storytelling more engaging and fun for children. We all know how much kids love stories, but it can be a challenge for parents and teachers to keep up with their constant demand for new ones. That's where our project comes in – it provides a solution to this problem by offering a vast library of pre-made stories on a variety of topics, as well as the ability to generate new stories based on children's interests.

But that's not all – we've also added an exciting new feature to our app that allows children to ask follow-up questions about the stories. The app will then narrate the answers, creating an interactive experience that encourages curiosity and engagement.

Our interactive story teller is designed to be incredibly user-friendly, with a simple interface that children can easily navigate. Whether they want to browse through pre-made stories or generate a new one based on their interests, the app provides a seamless experience. And with clear, engaging narration and accompanying visuals, children can fully immerse themselves in the story.

> **The Project consists of two repos, one for back-end and other for front-end. This repo consists of the code and installation steps for the back-end. For details on the frontend, please refer to the respective [repo](https://github.com/Aashray446/storyWizard)**

## Tech-stack

#### Frontend:

- ReactJS
- Tailwind CSS

#### Backend:

- Python
- Flask
- GPT, Stable Diffussion, Google text-to-speech API

## Installation Instructions

Run the following commands in the given order to get the front end running.

- `git clone https://github.com/gauthamk02/storyWizard-backend`
- `python -m venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- Generate your API key for OPENAI from [here](https://platform.openai.com/account/api-keys) and StabilityAI from [here](https://beta.dreamstudio.ai/account)
Add the API keys to the .env file in the following format

```
OPENAI_API_KEY=<key>
STABILITYAI_API_KEY=<key>
```

- Generate the google cloud credentials file from [here](https://cloud.google.com/text-to-speech/docs/before-you-begin) and save the JSON file as `service.json` in the root of the repository.

## Libraries and Dependencies

- Flask
- Flask CORS
- Pandas
- OpenAI
- Google cloud text to speech
- requests
- base64