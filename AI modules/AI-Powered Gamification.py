import cv2
from deepface import DeepFace
import numpy as np
import random

class GamificationAI:
    def __init__(self):
        self.emotion_history = []
        self.performance_history = []
        self.challenge_pool = {
            "easy": ["Solve 5 math problems", "Read a short story", "Answer 3 trivia questions"],
            "medium": ["Solve 10 math problems", "Write a paragraph", "Complete a puzzle"],
            "hard": ["Solve 20 math problems", "Write an essay", "Complete a complex puzzle"]
        }

    def detect_emotion(self):
        """
        Detect the user's emotion using facial recognition.
        
        :return: Detected emotion (e.g., "happy", "sad", "neutral").
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return None

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return None

        try:
            # Analyze the frame for emotions
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            emotion = result[0]["dominant_emotion"]
            print(f"Detected emotion: {emotion}")
            return emotion
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return None
        finally:
            cap.release()

    def adjust_lesson_style(self, emotion):
        """
        Adjust the lesson style based on the detected emotion.
        
        :param emotion: Detected emotion (e.g., "happy", "sad", "neutral").
        :return: Adjusted lesson style (e.g., "interactive", "relaxed", "challenging").
        """
        if emotion in ["happy", "surprise"]:
            return "interactive"
        elif emotion in ["sad", "angry"]:
            return "relaxed"
        else:
            return "challenging"

    def generate_challenge(self, performance):
        """
        Generate a custom challenge based on user performance.
        
        :param performance: User's performance score (0-100).
        :return: A dynamically generated challenge.
        """
        if performance < 30:
            difficulty = "easy"
        elif 30 <= performance < 70:
            difficulty = "medium"
        else:
            difficulty = "hard"

        challenge = random.choice(self.challenge_pool[difficulty])
        print(f"Generated {difficulty} challenge: {challenge}")
        return challenge

    def gamify_learning(self):
        """
        Main function to gamify the learning experience.
        """
        print("Welcome to AI-Powered Gamification!")
        while True:
            # Step 1: Detect emotion
            emotion = self.detect_emotion()
            if emotion:
                self.emotion_history.append(emotion)
                lesson_style = self.adjust_lesson_style(emotion)
                print(f"Adjusted lesson style: {lesson_style}")

            # Step 2: Simulate performance (for demonstration)
            performance = random.randint(0, 100)
            self.performance_history.append(performance)
            print(f"User performance: {performance}")

            # Step 3: Generate a challenge
            challenge = self.generate_challenge(performance)
            print(f"Your challenge: {challenge}")

            # Step 4: Wait for user input to continue
            input("Press Enter to continue or 'q' to quit...")
            if input().lower() == 'q':
                break

# Example usage
if __name__ == "__main__":
    ai = GamificationAI()
    ai.gamify_learning()
