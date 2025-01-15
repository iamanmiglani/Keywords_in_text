# # Keyword and CTA database
# keywords = {
#     "buy": ["go purchase", "go order now", "go acquire"],
#     "subscribe": ["check now", "get it for yourself", "enroll"],
#     "call to action": ["buy now", "click here", "get yours"]
# }


from transformers import AutoTokenizer, AutoModelForCausalLM
import timeit
import json
import os

# Load Falcon model and tokenizer
def load_falcon_model():
    model_name = "tiiuae/falcon-7b-instruct"  # Open-source Falcon model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    return tokenizer, model

tokenizer, model = load_falcon_model()

# Function to generate a response using Falcon
def generate_falcon_response(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    start = timeit.default_timer()
    outputs = model.generate(
        inputs.input_ids,
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
    )
    duration = timeit.default_timer() - start
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, duration

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
    # Keywords and call-to-action phrases
    keywords = {
        "buy": ["purchase", "order", "acquire"],
        "subscribe": ["sign up", "join", "enroll"],
        "call to action": ["buy now", "click here", "sign up today"]
    }

    # Load transcription
    transcription = process_transcription_file(transcription_file)

    # Prepare Falcon prompt
    prompt = f"""
    Analyze the following transcription for keywords and call-to-action phrases:
    {json.dumps(keywords)}
    Text: {transcription}
    """

    # Generate Falcon response
    response, duration = generate_falcon_response(prompt, tokenizer, model)

    # Search for keywords
    matches = search_keywords(transcription, keywords)

    # Output results
    print(f"Falcon Response Time: {duration:.2f} seconds\n")
    print("Keyword Matches:")
    for keyword, data in matches.items():
        print(f"- {keyword}: {data['count']} match(es) ({', '.join(data['matches'])})")

    # Save results to file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(f"Falcon Response Time: {duration:.2f} seconds\n\n")
        file.write("Keyword Matches:\n")
        for keyword, data in matches.items():
            file.write(f"- {keyword}: {data['count']} match(es) ({', '.join(data['matches'])})\n")
        file.write("\nFalcon Output:\n")
        file.write(response)

# Entry point
if __name__ == "__main__":
    transcription_path = "transcription.txt"  # Path to your transcription file
    output_path = "results.txt"  # Path to save the results
    main(transcription_path, output_path)
