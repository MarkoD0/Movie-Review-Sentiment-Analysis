# Functions for cleaning and preprocessing IMDB review text.
#
# Functions from this file are called in 2_preprocessing.ipynb and predict_review.py.
#
#
# The file includes:
#
#   Cleaning the entire dataset (basic_clean function), which includes:
#       1. Removing HTML (must be done first, because the HTML-encoded apostrophe "&#39;" must become a real apostrophe before we recognize contractions, and HTML tags attached to words would break tokenization if we left them for later)
#       2. Lowercase (must be done before expanding contractions, because the contraction dictionary is written in lowercase; otherwise, contractions at the beginning of a sentence would be missed)
#       3. Expanding contractions (done before removing punctuation (`don't` → `do not`, including forms without apostrophes such as `isnt` → `is not`, as well as some informal forms such as `gonna` → `going to`))
#       4. Removing numbers and punctuation
#       5. Tokenization (the cleaned text is now split by spaces)
#       6. Removing stop words (custom list that preserves negations; must be done after tokenization because it operates on individual cleaned words)
#       7. Removing tokens shorter than 2 characters
#
#   Four parallel variants:
#       1. Baseline     - without normalization (baseline_tokens function)
#       2. Stemming     - Snowball stemmer (stem_tokens function)
#       3. Lemmatization - WordNet lemmatizer with POS tagging (lemmatize_tokens function)
#       4. Lancaster    - Lancaster stemmer (lancaster_stem_tokens function)
#
#
# Functions from this file that are called in other files:
#   1. init_preprocessing       - downloads NLTK resources, builds the custom stop-word list, and prints the number of words in the stop-word list
#   2. basic_clean              - shared cleaning pipeline (the same for both processing branches: stemming and lemmatization)
#   3. baseline_tokens          - returns a joined string of tokens without normalization
#   4. stem_tokens              - applies the Snowball (Porter2) stemmer to the list of tokens, returns a joined string
#   5. lemmatize_tokens         - applies WordNet lemmatization with POS tagging to the list of tokens, returns a joined string
#   6. lancaster_stem_tokens    - applies the Lancaster stemmer to the list of tokens, returns a joined string
#
#
# Note: Some functions (e.g. _build_stopwords, _expand_contractions, _clean_html, _remove_non_alpha) are private and are used only within this file, but are not exposed externally because they are not needed by other parts of the code.


import re
import html as html_lib

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer, LancasterStemmer
from nltk import pos_tag



# ---------------------------------------------------------------------------
# Downloading NLTK (Natural Language Toolkit)
# ---------------------------------------------------------------------------
#
# All NLTK resources required for further text processing are downloaded: stop words, WordNet for lemmatization, and models for POS tagging (the name differs between library versions, so both names are listed)

def _download_nltk_resources() -> None:
    resources = [
        "stopwords",                        # List of common English "empty" words (the, is, a...)
        "wordnet",                          # Lexical database of the English language — WordNetLemmatizer uses it to find the base form of a word
        "omw-1.4",                          # Additional database (Open Multilingual Wordnet) required by newer NLTK versions for WordNet to work correctly
        "averaged_perceptron_tagger",       # Trained model for POS tagging (older resource name)
        "averaged_perceptron_tagger_eng",   # Trained model for POS tagging (newer resource name)   (naming scheme changed starting with version 3.9)
    ]
    for resource in resources:
        try:
            if nltk.download(resource, quiet=True):
                print(f"{resource}: OK")
            else:
                print(f"{resource}: FAILED")
        except Exception:
            print(f"{resource}: ERROR")



# ---------------------------------------------------------------------------
# Stop words (custom list that preserves negations)
# ---------------------------------------------------------------------------
#
# Standard NLTK English list, with words that carry negation removed
# Negation words are intentionally kept in the text because they carry a significant sentiment signal (e.g. "not good" and "good" are completely different)
# Builds a custom stop-word list: the standard NLTK English list minus "_NEGATION_WORDS" and returns it as a set
# The result is cached in the global variable "_stopwords_cache" so it does not need to be loaded again from NLTK resources

_NEGATION_WORDS = {"not", "no", "nor", "never", "neither"}

_stopwords_cache: set[str] | None = None

def _build_stopwords() -> set[str]:
    global _stopwords_cache
    if _stopwords_cache is None:
        try:
            base = set(nltk_stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            base = set(nltk_stopwords.words("english"))
        _stopwords_cache = base.difference(_NEGATION_WORDS)
    return _stopwords_cache



# ---------------------------------------------------------------------------
# Contraction dictionary
# ---------------------------------------------------------------------------
#
# Irregular contractions that cannot be generated by the general rule "n't → not" (e.g. "won't → will" not "wo not")
# Non-negation contractions that are also expanded for consistency ("it's → it is")
# Contractions written without apostrophes ("isnt", "dont", "wasnt"...) - a common informal variant in reviews that the general "n't" rule (which requires an apostrophe) would miss
# Several informal expressions common in more casually written reviews ("gonna", "kinda"...)
# (Intentionally, words such as "were" ("Actors *were* amazing") or "its" ("the movie and *its* plot") are not added to the no-apostrophe list - these words have legitimate alternative meanings in English.)

_IRREGULAR_CONTRACTIONS = {
    "won't": "will not", "can't": "can not", "cannot": "can not", "shan't": "shall not", "ain't": "is not",
    "let's": "let us", "that's": "that is", "there's": "there is",
    "here's": "here is", "what's": "what is", "who's": "who is",
    "how's": "how is", "when's": "when is", "where's": "where is", "why's": "why is",
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is", "it's": "it is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have", "what've": "what have", "who've": "who have", "where've": "where have", "when've": "when have", "why've": "why have",
    "i'll": "i will", "you'll": "you will", "he'll": "he will", "she'll": "she will", "it'll": "it will", "we'll": "we will", "they'll": "they will", "what'll": "what will", "who'll": "who will", "where'll": "where will", "when'll": "when will", "why'll": "why will",
    "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would", "it'd": "it would", "we'd": "we would", "they'd": "they would", "what'd": "what would", "who'd": "who would", "where'd": "where would", "when'd": "when would", "why'd": "why would", "how'd": "how did",
    "what're": "what are", "who're": "who are", "where're": "where are", "how're": "how are",
    "could've": "could have", "should've": "should have", "would've": "would have", "might've": "might have", "must've": "must have",
    "couldn't've": "could not have", "shouldn't've": "should not have", "wouldn't've": "would not have", "mightn't've": "might not have", "mustn't've": "must not have",
    # No-apostrophe variants of negation contractions
    "wont": "will not", "cant": "can not", "shant": "shall not", "aint": "is not",
    "dont": "do not", "doesnt": "does not", "didnt": "did not", "isnt": "is not",
    "arent": "are not", "wasnt": "was not", "werent": "were not", "havent": "have not",
    "hasnt": "has not", "hadnt": "had not", "couldnt": "could not", "wouldnt": "would not",
    "shouldnt": "should not", "mustnt": "must not", "neednt": "need not", "mightnt": "might not", "darent": "dare not",
    "theres": "there is", "heres": "here is", "whats": "what is", "thats": "that is", "whos": "who is", "hows": "how is", "wheres": "where is",
    "youve": "you have", "theyve": "they have", "youll": "you will", "theyll": "they will", "hes": "he is", "shes": "she is",
    "couldve": "could have", "shouldve": "should have", "wouldve": "would have", "mightve": "might have", "mustve": "must have", "theyre": "they are",
    "couldntve": "could not have", "shouldntve": "should not have", "wouldntve": "would not have", "mightntve": "might not have", "mustntve": "must not have", 
    # Informal expressions (no conflict with other meanings, safe to expand)
    "gonna": "going to", "wanna": "want to", "gotta": "got to", "kinda": "kind of", "sorta": "sort of", "outta": "out of", "tryna": "trying to",
    "imma": "i am going to", "lemme": "let me", "gimme": "give me", "dunno": "do not know", "c'mon": "come on", "comeon": "come on", "y'all": "you all",
    "coulda": "could have", "woulda": "would have", "shoulda": "should have", "musta": "must have", "mighta": "might have",
}

# A single regex that recognizes irregular contractions from the _IRREGULAR_CONTRACTIONS dictionary in one pass through the text
_IRREGULAR_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in _IRREGULAR_CONTRACTIONS.keys()) + r")\b")

# General rule for all remaining regular "n't" contractions with an apostrophe (e.g. "mightn't", which is not in the dictionary above)
_REGULAR_NEGATION_PATTERN = re.compile(r"\b(\w+)n't\b")

# Expands contractions in the text using the dictionary first and then the general rules (the text is expected to already be lowercase)
def _expand_contractions(text: str) -> str:
    text = _IRREGULAR_PATTERN.sub(lambda m: _IRREGULAR_CONTRACTIONS[m.group(0)], text)
    text = _REGULAR_NEGATION_PATTERN.sub(r"\1 not", text)
    return text



# ---------------------------------------------------------------------------
# HTML cleaning and remaining punctuation
# ---------------------------------------------------------------------------
#
# Decoding HTML entities and removing HTML tags (before expanding contractions)
# Removing numbers, punctuation, and excessive whitespace, while retaining only letters and spaces (after expanding contractions)

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")      # General HTML tag pattern (<br>, <i>, ...)
_NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")    # Any single character that is not a lowercase letter or whitespace (with .sub(" ", text), each such character becomes a space)
_MULTI_SPACE_PATTERN = re.compile(r"\s+")       # One or more consecutive spaces/tabs/newlines become a single space

def _clean_html(text: str) -> str:
    # Decodes HTML entities (e.g. &amp; -> &, &#39; -> ') using the built-in html module, and then removes ALL HTML tags (not just <br />, but generally any tag of the form <...>)
    text = html_lib.unescape(text)
    text = _HTML_TAG_PATTERN.sub(" ", text)
    return text

def _remove_non_alpha(text: str) -> str:
    # Removes numbers and punctuation - retains only letters and spaces
    text = _NON_ALPHA_PATTERN.sub(" ", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text



# ---------------------------------------------------------------------------
# Preprocessing initialization
# ---------------------------------------------------------------------------
#
# Encapsulates initialization functions for easier calling
# Downloads NLTK resources, builds a custom stop-word list, and prints the number of words in the stop-word list

def init_preprocessing() -> None:
    """
    Downloads required NLTK resources and builds the custom stopword list.
    """
    _download_nltk_resources()
    print("NLTK resources downloaded.")

    stopwords_set = _build_stopwords()
    print("\nCustom stopword list built.")
    print(f"Number of stopwords in the custom list: {len(stopwords_set)}")



# ---------------------------------------------------------------------------
# Shared cleaning pipeline
# ---------------------------------------------------------------------------
#
# Encapsulates the previously written functions for complete text cleaning and applies them before branching into the processing methods (stemming and lemmatization)
# The function returns a list of tokens that have not yet been stemmed or lemmatized
# The branching into stemming and lemmatization is performed after this function

def basic_clean(text: str) -> list[str]:
    """
    Cleans and tokenizes text by removing HTML, contractions, punctuation, and stopwords.
    Returns a list of cleaned tokens.
    """
    text = _clean_html(text)                                    # decoding HTML entities and removing HTML tags
    text = text.lower()                                         # converting to lowercase
    text = _expand_contractions(text)                           # expanding contractions
    text = _remove_non_alpha(text)                              # removing numbers and punctuation

    stopwords_set = _build_stopwords()                          # building the stopword set (already cached, so it is not loaded from NLTK resources every time; it is here so that the function is self-contained and does not depend on global state)
    tokens = text.split()                                       # tokenization
    tokens = [t for t in tokens if t not in stopwords_set]      # removing stopwords
    tokens = [t for t in tokens if len(t) >= 2]                 # removing tokens shorter than 2 characters

    return tokens





# ---------------------------------------------------------------------------
# Branch A: Baseline (without normalization)
# ---------------------------------------------------------------------------
#
# Control variant - uses the same cleaned and tokenized words as the following branches (after `basic_clean`: HTML, lowercase, contractions, numbers/punctuation, stop words, short tokens), but without stemming or lemmatization.
# The goal of this branch is to measure whether normalization (reducing words to their stem/lemma) contributes to model performance at all, or whether the models perform equally well on raw (but cleaned) tokens.
# Without this branch, we would not have a control point for comparison - we would only know whether stemming is better than lemmatization, but not whether either of them is better than omitting normalization.
# Returns a joined string of tokens

def baseline_tokens(tokens: list[str]) -> str:
    """
    Returns a string of tokens without any normalization.
    """
    return " ".join(tokens)



# ---------------------------------------------------------------------------
# Branch B: Stemming (Snowball / Porter2)
# ---------------------------------------------------------------------------
#
# Stemming is the process of roughly reducing words to their root by removing suffixes, based on predefined rules, without understanding grammar or context.
# The goal is to reduce different forms of the same word (e.g., "boring", "bored", "bore") to the same form, thereby reducing the vocabulary size and making it easier for the model to recognize that these words carry similar meaning.
# The result is often not a real English word (e.g., "movie" → "movi"), which is expected and does not represent a problem for the model.

_snowball = SnowballStemmer(language="english")

# Vraća spojeni string sa stemovanim tokenima (Snowball/Porter2)
def stem_tokens(tokens: list[str]) -> str:
    """
    Applies Snowball stemming to a list of tokens.
    Returns a string of stemmed words.
    """
    return " ".join(_snowball.stem(t) for t in tokens)



# ---------------------------------------------------------------------------
# Branch C: Lemmatization (WordNet + POS tagging)
# ---------------------------------------------------------------------------
#
# Lemmatization reduces a word to its grammatically correct base (dictionary) form — the lemma.
# It works using a lexical database (WordNet) and information about the part of speech (noun, verb, adjective, adverb).
# Lemmatization is slower but more precise: e.g. "running" correctly becomes "run", whereas with rougher methods it often remains unchanged.
# The result is always a real English word.
# We use POS tagging before lemmatization — without it, `WordNetLemmatizer` would assume that every word is a noun (e.g. "running" would not be reduced to "run").

_lemmatizer = WordNetLemmatizer()

# Returns the WordNet tag based on the Penn Treebank POS tag (WordNet lemmatization requires a POS tag to correctly determine the base form of the word)
def _get_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return "a"                      # adjective
    elif treebank_tag.startswith("V"):
        return "v"                      # verb
    elif treebank_tag.startswith("R"):
        return "r"                      # adverb
    else:
        return "n"                      # noun (default)

# Returns a joined string with lemmatized tokens (WordNet lemmatization + POS tagging)
def lemmatize_tokens(tokens: list[str]) -> str:
    """
    Applies WordNet lemmatization with POS tagging to a list of tokens.
    Returns a string of lemmatized words.
    """
    if not tokens:
        return ""
    tagged = pos_tag(tokens)
    lemmas = [
        _lemmatizer.lemmatize(token, pos=_get_wordnet_pos(tag))
        for token, tag in tagged
    ]
    return " ".join(lemmas)



# ---------------------------------------------------------------------------
# Branch D: Aggressive stemming (Lancaster)
# ---------------------------------------------------------------------------
#
# The Lancaster stemmer is known to be significantly more aggressive than Snowball (Branch B)
# It uses a smaller and rougher set of rules, often produces shorter roots, and merges more different words into the same form (e.g., words that Snowball leaves separate may be reduced to the same root by Lancaster).
# Motivation for this branch:
#   - Since we observed that (moderate) Snowball stemming gives somewhat better results than lemmatization, we test whether this trend "less grammatical precision → better result" is consistent with more aggressive stemming as well
#   - Or whether there is a point beyond which excessive aggressiveness starts to hurt (distinctions between terms with different meanings are lost)

_lancaster = LancasterStemmer()

# Returns a joined string with stemmed tokens (Lancaster)
def lancaster_stem_tokens(tokens: list[str]) -> str:
    """
    Applies Lancaster stemming to a list of tokens.
    Returns a string of lancaster stemmed words.
    """
    return " ".join(_lancaster.stem(t) for t in tokens)
