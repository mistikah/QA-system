import wikipediaapi
from transformers import pipeline

# Specify a user agent for Wikipedia API
user_agent = "PythonProject1/1.0 (https://github.com/mistikah/; annahmistikah@gmail.com) WikipediaAPI"

# Initialize Wikipedia with the user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent=user_agent
)

# Initialize the HuggingFace question-answering model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")


# Function to get Wikipedia summary
def get_wikipedia_summary(topic):
    page = wiki_wiki.page(topic)
    if page.exists():
        return page.summary
    else:
        return f"Sorry, I couldn't find information about '{topic}' on Wikipedia."


# Function to handle user questions
def answer_question(context, question):
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        return f"Error in answering the question: {e}"


# Main function to allow user interaction
def main():
    print("Welcome to the Question-Answer Bot!")
    print("Ask me a question, and I'll try to answer it using Wikipedia and AI.")
    print("Type 'exit' to quit.")

    while True:
        topic = input("\nEnter a topic you'd like to know about (or type 'exit' to quit): ")
        if topic.lower() == "exit":
            print("Goodbye!")
            break

        print("\nFetching information from Wikipedia...")
        context = get_wikipedia_summary(topic)

        if "couldn't find information" in context:
            print(context)
            continue

        print("\nWikipedia Summary:")
        print(context[:500] + ("..." if len(context) > 500 else ""))  # Show the first 500 characters of the summary

        question = input("\nWhat would you like to ask about this topic? (or type 'skip' to choose another topic): ")
        if question.lower() == "skip":
            continue

        print("\nThinking...")
        answer = answer_question(context, question)
        print("\nAnswer:", answer)


# Run the program
if __name__ == "__main__":
    main()
