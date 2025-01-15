

from llama_cpp import Llama
import timeit
import json
import os

# Load the LLaMA 2 model
llm = Llama(model_path="llama-2-7b-chat.ggmlv3.q2_K.bin", n_ctx=512, n_batch=128)

# Keyword and CTA database
keywords = {
    "buy": ["go purchase", "go order now", "go acquire"],
    "subscribe": ["check now", "get it for yourself", "enroll"],
    "call to action": ["buy now", "click here", "get yours"]
}

# Function to generate a response using LLaMA
def generate_llm_response(prompt, model):
    start = timeit.default_timer()
    output = model(
        prompt,
        max_tokens=512,
        echo=False,
        temperature=0.1,
        top_p=0.9,
    )
    duration = timeit.default_timer() - start
    return output['choices'][0]['text'], duration

# File handling for transcription
def process_transcription_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Search for keywords and call-to-action phrases
def search_keywords(text, keywords_dict):
    matches = {}
    for keyword, synonyms in keywords_dict.items():
        matches[keyword] = {
            "count": 0,
            "matches": []
        }
        for synonym in [keyword] + synonyms:
            count = text.lower().count(synonym.lower())
            if count > 0:
                matches[keyword]["count"] += count
                matches[keyword]["matches"].append(synonym)
    return matches

# Main function
def main(transcription_file, output_file):
    # Load transcription
    transcription = process_transcription_file(transcription_file)
    
    # Prepare LLaMA prompt
    prompt = f"""
    Analyze the following transcription for keywords and call-to-action phrases:
    {json.dumps(keywords)}
    Text: {transcription}
    """

    # Generate LLaMA response
    response, duration = generate_llm_response(prompt, llm)

    # Search for keywords
    matches = search_keywords(transcription, keywords)

    # Output results
    print(f"LLaMA Response Time: {duration:.2f} seconds\n")
    print("Keyword Matches:")
    for keyword, data in matches.items():
        print(f"- {keyword}: {data['count']} match(es) ({', '.join(data['matches'])})")

    # Save results to file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"LLaMA Response Time: {duration:.2f} seconds\n\n")
        file.write("Keyword Matches:\n")
        for keyword, data in matches.items():
            file.write(f"- {keyword}: {data['count']} match(es) ({', '.join(data['matches'])})\n")
        file.write("\nLLaMA Output:\n")
        file.write(response)

# Entry point
if __name__ == "__main__":
    transcription_path = "transcription.txt"  # Path to your transcription file
    output_path = "results.txt"  # Path to save the results
    main(transcription_path, output_path)
