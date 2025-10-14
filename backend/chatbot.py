import json
import random
import re
import os
import numpy as np
import sys
import torch
# --- MODIFIED: Import Flair instead of spaCy ---
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# --- NEW: Import SentenceTransformer ---
from sentence_transformers import SentenceTransformer, util

from flair.data import Sentence
from flair.models import SequenceTagger
import wikipediaapi


class Chatbot:
    # --- MODIFIED: Changed model_file to embedding_file ---
    def __init__(self, intents_file, embedding_file='intent_embeddings.pt', debug=False):
        self.debug = debug
        self.embedding_file = embedding_file
        self.intents_file_path = intents_file  # Store the path for later comparison
        
        # --- NEW: Load the Sentence Transformer model ---
        # This model is multilingual and great for semantic similarity.
        print("[INFO] Loading Sentence Transformer model ('paraphrase-multilingual-MiniLM-L12-v2')...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("[INFO] Sentence Transformer model loaded.")
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        # --- NEW: Map Flair's tags to our internal placeholder names ---
        self.entity_map = {
            'PER': 'PERSON',  # Person
            'LOC': 'GPE',     # Location
            'GPE': 'GPE',     # Geo-Political Entity
            'ORG': 'ORG'      # Organization
        }

        # --- NEW: Load Flair NER model for Indonesian ---
        try:
            print("[INFO] Loading Flair NER model for Indonesian ('id-ner'). This may take a moment...")
            self.ner_tagger = SequenceTagger.load('id-ner')
            print("[INFO] Flair NER model loaded successfully.")
        except Exception as e:
            print(f"\n[ERROR] Failed to load Flair NER model. Error: {e}")
            print("Please ensure you have an internet connection for the first download.")
            self.ner_tagger = None
        
        # --- MODIFIED: Use Lazy Initialization for Wikipedia API ---
        self.wiki_api = None # Will be initialized on first use
        self.intents = self.load_intents(intents_file)

        # --- NEW: Attributes for the new model ---
        self.corpus_embeddings = None
        self.corpus_tags = None

        # --- NEW: Add state for conversation context ---
        self.last_entities = {}
        if self.intents:
            print(f"[INFO] Successfully loaded {len(self.intents)} intents from {os.path.basename(intents_file)}.")
            # --- MODIFIED: Call the new pre-computation method ---
            self.load_or_precompute_embeddings()
        else:
            print(f"[WARNING] No intents were loaded. The bot will only give default responses.")
            self.model = None

    def _preprocess_text(self, text):
        """Lowercase and stem the text. (Now less critical for intent matching)"""
        return self.stemmer.stem(text.lower())

    def load_intents(self, file_path):
        """Loads intents from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            # Check that the 'intents' key exists before trying to access it
            if 'intents' not in data:
                print(f"Error: JSON file at {file_path} is missing the 'intents' key.")
                return []
            return data.get('intents', [])
        except FileNotFoundError:
            print(f"\n--- CRITICAL ERROR ---")
            print(f"Error: The file '{os.path.basename(file_path)}' was not found in the expected directory.")
            dir_path = os.path.dirname(file_path)
            print(f"Expected directory: {dir_path}")
            if os.path.isdir(dir_path):
                print(f"Files currently in this directory: {os.listdir(dir_path)}")
            print(f"Please ensure the file exists and the name is spelled correctly.\n")
            return []
        except json.JSONDecodeError:
            print(f"Error: The file at {file_path} is not a valid JSON file.")
            return []

    # --- MODIFIED: This method now pre-computes embeddings instead of training a classifier ---
    def load_or_precompute_embeddings(self):
        """Loads pre-computed embeddings, or creates them if they don't exist or if intents.json is newer."""
        embeddings_exist = os.path.exists(self.embedding_file)
        recompute_needed = False

        if embeddings_exist:
            # Check if intents.json has been modified since the model was last created
            embedding_mod_time = os.path.getmtime(self.embedding_file)
            intents_mod_time = os.path.getmtime(self.intents_file_path)
            if intents_mod_time > embedding_mod_time:
                recompute_needed = True
                print("[INFO] 'intents.json' has been updated. Re-computing embeddings...")

        if embeddings_exist and not recompute_needed:
            print(f"[INFO] Loading pre-computed embeddings from {self.embedding_file}...")
            data = torch.load(self.embedding_file)
            self.corpus_tags = data['tags']
            self.corpus_embeddings = data['embeddings']
            print("[INFO] Embeddings loaded successfully.")
        else:
            if not embeddings_exist:
                print(f"[INFO] No embedding file found. Pre-computing for the first time...")
            self._precompute_and_save_embeddings()

    # --- NEW: This is the new "training" step ---
    def _precompute_and_save_embeddings(self):
        """Encodes all patterns from intents.json into embeddings and saves them."""
        if not self.intents:
            print("[ERROR] Cannot compute embeddings without intents.")
            return

        corpus_sentences = []
        self.corpus_tags = []
        for intent in self.intents:
            if 'patterns' in intent:
                for pattern in intent['patterns']:
                    # IMPORTANT: We use the raw pattern, not the stemmed one, for better semantic meaning.
                    corpus_sentences.append(pattern)
                    self.corpus_tags.append(intent['tag'])

        if not corpus_sentences:
            print("[ERROR] No patterns found in intents file. Cannot compute embeddings.")
            return

        print(f"[INFO] Encoding {len(corpus_sentences)} patterns. This might take a moment...")
        self.corpus_embeddings = self.model.encode(corpus_sentences, convert_to_tensor=True)
        torch.save({'tags': self.corpus_tags, 'embeddings': self.corpus_embeddings}, self.embedding_file)
        print(f"[INFO] Embeddings computed and saved to {self.embedding_file}.")

    def _get_fallback_response(self):
        """Finds and returns a random fallback response."""
        for intent in self.intents:
            if intent['tag'] == 'fallback':
                return random.choice(intent['responses']), 'fallback'
        # A secondary, hardcoded fallback in case the 'fallback' intent is missing from JSON
        return "Maaf, saya tidak mengerti maksud Anda.", 'fallback'

    def _extract_entities(self, message):
        """Extracts named entities from a message using Flair."""
        if not self.ner_tagger:
            return {}
        
        # Create a Flair Sentence object. Using .title() can improve
        # NER accuracy for proper nouns like names and locations.
        sentence = Sentence(message.title())
        
        # Predict NER tags
        self.ner_tagger.predict(sentence)
        
        entities = {}
        # Iterate over entities and extract them
        for entity in sentence.get_spans('ner'):
            flair_tag = entity.tag
            # We use the entity's tag (e.g., 'PER' for person) and map it
            # to our internal placeholder name (e.g., 'PERSON').
            if flair_tag in self.entity_map:
                internal_tag = self.entity_map[flair_tag]
                # This will overwrite if multiple entities of the same type are found.
                entities[internal_tag] = entity.text
            
        if self.debug and entities:
            print(f"[DEBUG] Entities found: {entities}")
        return entities

    def _format_response_with_entities(self, response_list, entities):
        """
        Finds the best response from a list, prioritizing templates that can be
        fully formatted with the found entities.
        """
        generic_responses = [r for r in response_list if '{' not in r]
        templated_responses = [r for r in response_list if '{' in r]

        if entities and templated_responses:
            possible_responses = []
            # Find all templates that can be successfully formatted
            for template in templated_responses:
                # Find all placeholders like {PERSON}, {GPE} in the template
                required_keys = set(re.findall(r'\{(\w+)\}', template))
                
                # Check if all required keys are present in our found entities
                if required_keys.issubset(entities.keys()):
                    possible_responses.append(template.format(**entities))
            
            # If we found any responses we could format, pick one randomly
            if possible_responses:
                return random.choice(possible_responses)

        # If no suitable template was found or no entities were provided, fall back to a generic response.
        return random.choice(generic_responses) if generic_responses else "Saya mengerti, tapi tidak bisa memberikan respons spesifik."

    def _get_wiki_api(self):
        """
        Initializes the Wikipedia API on first use (Lazy Initialization).
        This avoids startup conflicts and makes the bot more robust.
        """
        # If the API object already exists, just return it.
        if self.wiki_api:
            return self.wiki_api

        # If not, try to create it.
        try:
            print("[INFO] Initializing Wikipedia API for the first time...")
            api = wikipediaapi.Wikipedia(
                language='id',
                user_agent='MyChatbot/1.0 (rahmat@example.com)', # Good practice
                extract_format=wikipediaapi.ExtractFormat.WIKI
            )
            self.wiki_api = api # Store it for future use
            return self.wiki_api
        except Exception as e:
            print(f"\n[WARNING] Failed to initialize Wikipedia API. Error: {e}")
            # It will remain None, and subsequent calls will fail gracefully.
            return None

    def _fetch_wikipedia_summary(self, topic):
        """Fetches a short summary of a topic from Wikipedia."""
        wiki_api = self._get_wiki_api() # Get the API object
        if not wiki_api:
            return "Maaf, fitur pencarian Wikipedia tidak dapat diinisialisasi. Mohon periksa koneksi internet Anda."
        
        # --- NEW: Automatically capitalize the topic for better search results ---
        topic = topic.title()

        if self.debug:
            print(f"[DEBUG] Fetching Wikipedia summary for topic: '{topic}'")

        try:
            page = wiki_api.page(topic)
            
            if not page.exists():
                if self.debug:
                    print(f"[DEBUG] Wikipedia page for '{topic}' does not exist.")
                return f"Maaf, saya tidak dapat menemukan informasi tentang {topic} di Wikipedia."

            # Return the first paragraph of the summary
            summary = page.summary.split('\n')[0]
            # A quick check to see if the summary is empty or just the topic name
            if not summary or summary.lower().strip() == topic.lower().strip():
                return f"Saya menemukan halaman untuk {topic}, tapi tidak ada ringkasan yang tersedia."
            return summary
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] An error occurred during Wikipedia fetch: {e}")
            return "Maaf, terjadi kesalahan saat mencoba menghubungi Wikipedia. Silakan coba lagi nanti."

    def get_response(self, message):
        """Finds the most similar intent and returns a response."""
        if not self.model:
            return "Error: Chatbot model is not available.", "error"

        # --- NEW: Transformer-based prediction logic ---
        # 1. Encode the user's message into an embedding
        query_embedding = self.model.encode(message, convert_to_tensor=True)

        # 2. Compute cosine similarity between the query and all corpus patterns
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

        # 3. Find the most similar pattern
        top_result = torch.topk(cos_scores, k=1)
        
        # 4. Get the score and index of the best match
        score = top_result[0].item()
        idx = top_result[1].item()

        if self.debug:
            print(f"[DEBUG] Best match: '{self.corpus_tags[idx]}' with score {score:.4f}")

        # --- MODIFIED: Use a new confidence threshold suitable for cosine similarity ---
        CONFIDENCE_THRESHOLD = 0.5
        if score > CONFIDENCE_THRESHOLD:
            tag = self.corpus_tags[idx]
            if self.debug:
                print(f"[DEBUG] >> Predicted Tag: '{tag}' with confidence {score:.4f}")

            # Find the list of all possible responses for the predicted tag
            all_responses = []
            for intent in self.intents:
                if intent['tag'] == tag:
                    all_responses = intent.get('responses', [])
                    break
            
            if not all_responses:
                return self._get_fallback_response()

            # --- IMPROVED: Hybrid handling for 'about_person' intent ---
            if tag == 'about_person':
                topic = None
                
                # --- MODIFIED: Prioritize NER for higher accuracy ---
                # 1. Try to find a person using NER first.
                entities = self._extract_entities(message)
                if 'PERSON' in entities:
                    topic = entities['PERSON']
                    if self.debug: print(f"[DEBUG] Topic found via NER: '{topic}'")

                # 2. If NER fails, fall back to rule-based extraction.
                if not topic:
                    if self.debug: print("[DEBUG] NER failed, trying rule-based extraction...")
                    triggers = [
                        "siapa itu", "ceritakan tentang", "aku ingin tahu tentang", 
                        "saya ingin tahu tentang", "jelaskan tentang", "siapakah", "info tentang"
                    ]
                    lower_message = message.lower()
                    for trigger in triggers:
                        trigger_pos = lower_message.find(trigger)
                        if trigger_pos != -1:
                            topic = message[trigger_pos + len(trigger):].strip()
                            if self.debug: print(f"[DEBUG] Topic found via rule: '{topic}'")
                            break
                
                # 3. Final fallback for short messages if both methods fail.
                if not topic and len(message.split()) <= 4:
                    topic = message
                    if self.debug: print(f"[DEBUG] No topic found, assuming entire short message is the topic: '{topic}'")

                # 4. If we found a topic, clean it up and fetch from Wikipedia
                if topic:
                    # --- NEW: Clean the topic string from conversational fluff ---
                    junk_words = ['sih', 'dong', 'ya', 'kah', 'itu']
                    # This regex removes junk words and punctuation from the end of the string
                    pattern = r'(\s*\b(' + '|'.join(junk_words) + r')\b\s*|[?\.!])+$'
                    cleaned_topic = re.sub(pattern, '', topic, flags=re.IGNORECASE).strip()

                    if self.debug: print(f"[DEBUG] Original topic: '{topic}' -> Cleaned topic: '{cleaned_topic}'")

                    # Handle cases where the cleaned topic is a pronoun (contextual follow-up)
                    if cleaned_topic.lower() in ['dia', 'nya', 'beliau', 'tentangnya']:
                        if 'PERSON' in self.last_entities:
                            person_from_context = self.last_entities['PERSON']
                            wiki_summary = self._fetch_wikipedia_summary(person_from_context)
                            return wiki_summary, 'person_inquiry_followup'
                        else:
                            return "Dia siapa yang Anda maksud? Mohon sebutkan nama lengkapnya.", "clarification_needed"

                    self.last_entities['PERSON'] = cleaned_topic
                    wiki_summary = self._fetch_wikipedia_summary(cleaned_topic)
                    return wiki_summary, tag
            
            # --- NEW: Handling for contextual follow-up questions ---
            if tag == 'person_inquiry_followup':
                # Check if there is a person in our context memory
                if 'PERSON' in self.last_entities:
                    person_from_context = self.last_entities['PERSON']
                    wiki_summary = self._fetch_wikipedia_summary(person_from_context)
                    return wiki_summary, tag
                else:
                    # No context, so we ask for clarification
                    return "Dia siapa yang Anda maksud? Mohon sebutkan nama lengkapnya.", "clarification_needed"

            # --- Generic response handling for all other intents (and as a final fallback) ---
            entities = self._extract_entities(message) # NER for other intents like 'help'
            final_response = self._format_response_with_entities(all_responses, entities)
            return final_response, tag

        # If confidence is too low, use the structured fallback responses
        return self._get_fallback_response()

if __name__ == "__main__":
    # Build a robust path to the intents file, relative to the script's location.
    # This ensures the file is found regardless of the current working directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    intents_path = os.path.join(script_dir, 'intents.json')
    embedding_path = os.path.join(script_dir, 'intent_embeddings.pt')

    # Set debug=False to hide debug messages
    bot = Chatbot(intents_path, embedding_file=embedding_path, debug=True)

    # --- IMPROVEMENT: Fail-fast if the model is not ready ---
    # Check if the model was successfully loaded or trained. If not, exit the program.
    if not bot.model:
        print("\n[CRITICAL] Chatbot model could not be initialized. Exiting program.")
        print("Please check for errors above, such as a missing 'intents.json' file or invalid content.")
        sys.exit(1) # Exit with a non-zero status code to indicate an error

    print("\nBot: Halo! Saya adalah chatbot sederhana. Ketik 'selamat tinggal' untuk keluar.")
    chatting = True
    while chatting:
        user_input = input("You: ")
        response, tag = bot.get_response(user_input)
        print(f"Bot: {response}")
        # Stop the chat if the intent is to say goodbye or if a critical error occurred.
        if tag == "goodbye" or tag == "error":
            chatting = False