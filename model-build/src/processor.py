import nltk
import json

from .utils import remove_stop_words


class TextProcessor:
    def __init__(self):
        self.punctuation = "[\"'?,\.]"  # I will replace all these punctuation with ''

        # make the abbreviation dictionary
        self.abbr_dict = {
            "what's": "what is",
            "what're": "what are",
            "where's": "where is",
            "where're": "where are",
            "i'm": "i am",
            "we're": "we are",
            "it's": "it is",
            "that's": "that is",
            "there's": "there is",
            "there're": "there are",
            "i've": "i have",
            "who've": "who have",
            "would've": "would have",
            "not've": "not have",
            "i'll": "i will",
            "it'll": "it will",
            "isn't": "is not",
            "wasn't": "was not",
            "aren't": "are not",
            "weren't": "were not",
            "can't": "can not",
            "couldn't": "could not",
            "don't": "do not",
            "didn't": "did not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "doesn't": "does not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "won't": "will not",
            "you're": "you are",
            self.punctuation: "",
            "\s+": " ",  # replace multi space with one single space
        }

    def process_data(self, data):
        # Convert to lower case
        data.text = data.text.str.lower()

        # convert to string
        data.text = data.text.astype(str)

        # replace abbreviations
        data.text.replace(self.abbr_dict, regex=True, inplace=True)

        # remove stop words
        data.text = data.text.apply(remove_stop_words)

        # apply lemmatization
        lemmatizer = nltk.stem.WordNetLemmatizer()

        # process the text
        data["processed_text"] = data["text"].apply(
            lambda x: " ".join([lemmatizer.lemmatize(y) for y in x.split()])
        )

        return data
