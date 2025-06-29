import json
import random

# Load dataset once at startup
with open('./app/dataset.json', 'r') as f:
    dataset = json.load(f)

def get_youtube_iframe(question):
    matching_videos = []

    for item in dataset:
        if item["question"].lower() in question.lower():
            matching_videos.extend(item["video_ids"].split(","))

    if not matching_videos:
        matching_videos = ["dQw4w9WgXcQ"]

    selected_video = random.choice(matching_videos).strip()

    return f'<iframe width="300" height="200" src="https://www.youtube.com/embed/{selected_video}" frameborder="0"></iframe>'
