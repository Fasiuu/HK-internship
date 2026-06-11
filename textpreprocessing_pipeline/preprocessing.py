# preprocessing.py
import re
import string
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

# Configure Logging to track pipeline steps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextPipeline:
    def __init__(self, remove_numbers=True, use_logging=False):
        self.remove_numbers = remove_numbers
        self.use_logging = use_logging
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def _log(self, message):
        """Helper to print steps if logging is enabled."""
        if self.use_logging:
            logging.info(message)

    def clean_text(self, text: str) -> str:
        """Step 1: Lowercasing, stripping, and regex-based cleaning."""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and HTML tags (Good practice for edge cases)
        text = re.sub(r'https?://\s+|www\.\s+', '', text)
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation and special characters using regex
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Optional: Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
            
        # Remove extra whitespace/newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        self._log(f"Cleaned Text: '{text}'")
        return text

    def tokenize(self, text: str) -> list:
        """Step 2: Custom Tokenization using Regex."""
        # \b\w+\b finds all alphanumeric word blocks boundaries
        tokens = re.findall(r'\b\w+\b', text)
        self._log(f"Tokens: {tokens}")
        return tokens

    def remove_stopwords(self, tokens: list) -> list:
        """Step 3: Filter out standard English stopwords."""
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        self._log(f"Filtered Tokens (No Stopwords): {filtered_tokens}")
        return filtered_tokens

    def _get_wordnet_pos(self, word):
        """Map standard POS tag to a character WordNetLemmatizer accepts."""
        # Simple fallback logic or default to NOUN
        # For a production pipeline, you could use nltk.pos_tag([word])
        return wordnet.NOUN

    def lemmatize_tokens(self, tokens: list) -> list:
        """Step 4: Reduce words to their base semantic form."""
        # Advanced tip: For words like 'running', we explicitly tell it it's a Verb ('v')
        lemmatized = []
        for token in tokens:
            # Simple contextual check for common verbs ending in 'ing' or 'ed'
            pos = wordnet.VERB if token.endswith(('ing', 'ed')) else wordnet.NOUN
            lemmatized.append(self.lemmatizer.lemmatize(token, pos=pos))
            
        self._log(f"Lemmatized Tokens: {lemmatized}")
        return lemmatized

    def stem_tokens(self, tokens: list) -> list:
        """Bonus Step: Crude algorithmic stemming (Porter Stemmer)."""
        stemmed = [self.stemmer.stem(token) for token in tokens]
        return stemmed

    def process(self, text: str, method='lemma') -> list:
        """Combines all steps into a single execution pipeline."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
            
        tokens = self.tokenize(cleaned)
        tokens_no_stop = self.remove_stopwords(tokens)
        
        if method == 'stem':
            return self.stem_tokens(tokens_no_stop)
        return self.lemmatize_tokens(tokens_no_stop)