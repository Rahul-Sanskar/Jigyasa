import speech_recognition as sr
import spacy

nlp = spacy.load("en_core_web_sm")

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I didn't understand."
    except sr.RequestError:
        return "API unavailable."

def process_text(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop]

speech_text = recognize_speech()
print("Processed Words:", process_text(speech_text))
