import numpy as np
import nltk
import json


# custom modules
from .utils import model_loader, present_in_db, store_classifications
from .extractor import TextExtractor
from .processor import TextProcessor


class UrlClassifier:
    def __init__(self):
        self.extractor = TextExtractor()
        self.processor = TextProcessor()
        self.vectorizer = model_loader("vectorizer")
        self.model = model_loader("ml-model")

        self.class_mapping = {
            0: "Machine Learning ",
            1: "Data Structures and Algorithms",
            2: "Web Development",
            3: "Social Science",
            4: "Management",
        }

        # if self._first_start():

        #     # download required content
        #     nltk.download("wordnet")
        #     nltk.download("omw-1.4")
        #     nltk.download("stopwords")
        #     nltk.download("punkt")

        #     # set started as false
        #     started = {"start": False}
        #     json.dump(started, open("first_start.json", "w"))

    # need to check out this path variable
    def _first_start(self):
        return json.load(open("first_start.json", "r"))["start"]

    def _to_classify_urls(self, stored_urls, list_urls):
        """Return which urls are not stored in the dB

        Args:
            stored_urls (_type_): _description_
            list_urls (_type_): _description_

        Returns:
            _type_: _description_
        """
        to_classify_urls = []

        # form the urls to classify
        stored_url_set = set(stored_urls)

        for url in list_urls:
            if url not in stored_url_set:
                to_classify_urls.append(url)

        to_classify_urls = np.array(to_classify_urls)

        return to_classify_urls

    def _transform_data(self, dataframe):
        # transform after the processing
        test_vector = self.vectorizer.transform(dataframe["processed_text"])

        return test_vector

    def _get_predictions(self, data_vector):
        predicted = self.model.predict(data_vector)

        class_labels = []
        for label in predicted:
            class_labels.append(self.class_mapping[label])

        return class_labels

    def classify_urls(self, list_urls, debug=False, return_predictions=False):
        """Main function called by the predict API end point

        Args:
            list_urls (List[str]): list of urls requested by the history tracker
        """
        # list of urls
        list_urls = np.array(list_urls)

        stored_urls, stored_classes = present_in_db(list_urls)

        # get the array for classifications
        to_classify_urls = self._to_classify_urls(stored_urls, list_urls)

        # nothing to classify
        if len(to_classify_urls) == 0:
            if return_predictions:
                return stored_urls, stored_classes
            else:
                return

        # we will need the dataframe now, with extracted text
        dataframe = self.extractor.extract(to_classify_urls, debug=debug)

        # process your text
        processed_dataframe = self.processor.process_data(dataframe)

        # transform it
        transformed_vector = self._transform_data(processed_dataframe)

        # class predictions
        predictions = self._get_predictions(transformed_vector)

        # store these predictions with respective urls
        store_classifications(to_classify_urls, predictions)

        # debug info
        if debug:
            print("Stored : ")
            print(stored_urls)
            print(stored_classes)

            print("To classify :")
            print(to_classify_urls)
            print(predictions)

        if return_predictions:
            # store urls
            final_urls = to_classify_urls
            if len(stored_urls) > 0:
                final_urls += stored_urls

            # store the classes
            final_classes = predictions
            if len(stored_classes) > 0:
                final_classes += stored_classes

            return final_urls, final_classes
