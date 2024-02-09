import openai
proxy = {
'http': 'http://localhost:7890',
'https': 'http://localhost:7890'
}

openai.proxy = proxy

# Set up your OpenAI API credentials
openai.api_key = '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'

def rephrase_sentence(sentence):
    # Define the prompt
    prompt = f"Slightly adjust the following sentence: '{sentence}'"

    # Generate text using the completions API
    response = openai.Completion.create(
        engine='gpt-3.5-turbo-instruct',  # You can also use 'gpt-3.5-turbo' for faster responses
        prompt=prompt,
        max_tokens=50,  # You can adjust this based on the desired length of the response
        temperature=0.7,  # Controls the randomness of the output, lower values make it more focused
        n=1,  # Generate only one response
        stop=None,  # Let the model generate a full response without any specific stop sequence
        timeout = 1000
    )

    # Extract the rephrased sentence from the API response
    rephrased_sentence = response.choices[0].text.strip()

    return rephrased_sentence

# Call the function
sentence = "of an astronaut riding a horse on mars"
# sentence = "a photo"
rephrased = rephrase_sentence(sentence)
print(f"Original: {sentence}")
print(f"Rephrased: {rephrased}")