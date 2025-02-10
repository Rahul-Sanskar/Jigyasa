from transformers import pipeline

# Load a pre-trained text generation model
story_generator = pipeline("text-generation", model="gpt-2")

def generate_story(theme, child_name, learning_goal):
    prompt = f"Once upon a time, there was a child named {child_name} who loved {theme}. One day, they decided to learn about {learning_goal}. "
    story = story_generator(prompt, max_length=200, num_return_sequences=1)
    return story[0]["generated_text"]

# Example usage
theme = "space exploration"
child_name = "Alice"
learning_goal = "the solar system"
story = generate_story(theme, child_name, learning_goal)
print(story)


import spacy
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import random

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return list(set(keywords))

def generate_questions(text):
    keywords = extract_keywords(text)
    questions = []
    for keyword in keywords:
        question = f"What is {keyword}?"
        questions.append(question)
    return questions

# Example usage
questions = generate_questions(story)
for q in questions:
    print(q)
    
    
def interactive_story(theme, child_name, learning_goal):
    # Generate story
    story = generate_story(theme, child_name, learning_goal)
    print("Here's your story:\n", story)

    # Generate questions
    questions = generate_questions(story)
    print("\nLet's test your understanding! Answer these questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

# Example usage
interactive_story("ancient Egypt", "Bob", "pyramids")


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/generate_story", methods=["POST"])
def generate_story_api():
    data = request.json
    theme = data["theme"]
    child_name = data["child_name"]
    learning_goal = data["learning_goal"]
    story = generate_story(theme, child_name, learning_goal)
    return jsonify({"story": story})

@app.route("/generate_questions", methods=["POST"])
def generate_questions_api():
    data = request.json
    text = data["text"]
    questions = generate_questions(text)
    return jsonify({"questions": questions})

if __name__ == "__main__":
    app.run(debug=True)
    
    
