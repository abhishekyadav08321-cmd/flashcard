import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

MODEL = "openai/gpt-oss-120b"


def generate_flashcards(topic: str, number_of_cards: int) -> str:
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic cannot be empty.")
    if number_of_cards < 1:
        raise ValueError("Number of flashcards must be at least 1.")

    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("HF_TOKEN is not set. Add it to your .env file.")

    client = InferenceClient(api_key=api_key, provider="auto")
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

    if not response.choices:
        raise RuntimeError("The model returned no flashcards.")
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The model returned an empty response.")
    return content


def main() -> None:
    print("================================")
    print("      FLASHCARD GENERATOR")
    print("================================")

    topic = input("Enter topic: ").strip()
    if not topic:
        print("Topic cannot be empty.")
        return

    try:
        number = int(input("Number of flashcards: "))
    except ValueError:
        print("Please enter a whole number.")
        return

    if number < 1:
        print("Number of flashcards must be at least 1.")
        return

    print("\nGenerating flashcards...\n")
    print(generate_flashcards(topic, number))


if __name__ == "__main__":
    main()