# import
from trainers.base_trainer import SimpleTrainer
import time
import sys 
sys.path.append("..") 
import openai
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff




class Pic_LLMMixin(SimpleTrainer):
    
    def get_api_key(self):
        # api-key file-path
        file_path = '/home/wenhesun/Plum_Final/api-key.txt'
        output_file_path = '/home/wenhesun/Plum_Final/get_api-key.txt'
        # list which is used to restored the result
        matched_strings = []

        # Read the file by line
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Look for strings that start with "sk-" and are followed by 48-bit characters in each line
                matches = re.findall(r'sk-\w{48}', line)
                # Add the found matches to the list
                matched_strings.extend(matches)

        return matched_strings

    def gpt_rephraser(self, sentence):
        
        api_keys = self.get_api_key()
        proxy = {
            'http': 'http://localhost:7890',
            'https': 'http://localhost:7890'
        }
        # Set up a proxy
        openai.proxy = proxy  
        # Define the prompt
        prompt = f"Generate a random sentence piece with similar structures as: '{sentence}'"       
        for key in api_keys:
            try:
                # Set API key
                openai.api_key = key              
                # Use API key
                response = openai.Completion.create(
                    engine='gpt-3.5-turbo-instruct',
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.7,
                    n=1,
                    stop=None,
                    timeout=100
                )               
                # If successful, the rephrased sentence is extracted from the response and returned
                rephrased_sentence = response.choices[0].text.strip()
                return rephrased_sentence
            
            except Exception as e:
                # If the call fails, print the error message and continue trying the next API key
                # print(f"Error with API key {key}: {e}")
                continue
        
        # If all API keys fail to attempt, an error message is returned
        return "Failed to rephrase the sentence using all available API keys."


    def gpt_adjuster(self, sentence):
        api_keys = self.get_api_key()
        proxy = {
            'http': 'http://localhost:7890',
            'https': 'http://localhost:7890'
        }
        # Set up a proxy
        openai.proxy = proxy  
        # Define the prompt
        prompt = f"Rephrase and refine the following piece of sentence in more detail: '{sentence}'"
        # Generate text using the completions API
        for key in api_keys:
            try:
                # Set API key
                openai.api_key = key              
                # Use API key
                response = openai.Completion.create(
                    engine='gpt-3.5-turbo-instruct',
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.7,
                    n=1,
                    stop=None,
                    timeout=100
                )               
                # If successful, the rephrased sentence is extracted from the response and returned
                adjusted_sentence = response.choices[0].text.strip()
                return adjusted_sentence
            
            except Exception as e:
                # If the call fails, print the error message and continue trying the next API key
                # print(f"Error with API key {key}: {e}")
                continue
        
        # If all API keys fail to attempt, an error message is returned
        return "Failed to rephrase the sentence using all available API keys."


    def mutated(self, candidate, phrase_lookup, use_add, delete_tracker, edit_operations, args):
        # print("Running LLM mutation")
        deleted = {}
        added = {}
        candidate = self.gpt_rephraser(candidate)
        return candidate, deleted, added



            
