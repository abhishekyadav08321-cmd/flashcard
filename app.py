import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("hf_DVVEXWfVEgmvoawGmusRcsxLkyzwjLqIhSpy"),
    provider="auto"
)

MODEL = "openai/gpt-oss-120b"


def generate_flashcards(topic, number_of_cards):

    prompt = f"""
Create {number_of_cards} flashcards about {topic}.

Format:

Flashcard 1:
Question: ...
Answer: ...

Flashcard 2:
Question: ...
Answer: ...

Keep the questions and answers short,
clear and accurate.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an educational flashcard generator."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1000,
        temperature=0.7
    )

    return response.choices[0].message.content


print("================================")
print("      FLASHCARD GENERATOR")
print("================================")

topic = input("Enter topic: ")
number = int(input("Number of flashcards: "))

print("\nGenerating flashcards...\n")

result = generate_flashcards(topic, number)

print(result)