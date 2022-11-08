from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from .classifier import UrlClassifier


url_classifier = UrlClassifier()


# url_list = [
#     "https://www.computerhope.com/jargon/b/binary.htm",
#     "https://en.wikipedia.org/wiki/Algorithm",
#     "https://www.geeksforgeeks.org/introduction-to-algorithms/",
#     "https://www.vocabulary.com/dictionary/social",
#     "https://learn.g2.com/what-is-graphic-design",
#     "https://www.w3schools.com/REACT/DEFAULT.ASP",
#     "https://www.ibm.com/cloud/learn/machine-learning",
# ]


# print(url_classifier.classify_urls(url_list))


app = Flask(__name__)
CORS(app)


@app.route("/process_urls", methods=["GET", "POST"])
def process_and_store():
    """Takes in json as :

    json : {
        "url_list" : ["1", "2", ...]
    }

    Returns:
        _type_: _description_
    """
    url_list = request.args.get("url_list")

    url_classifier.classify_urls(url_list, debug=True, return_predictions=False)
    status = True

    return jsonify(status)


@app.route("/get_url_classes", methods=["GET", "POST"])
def get_url_classification():
    url_list = request.args.get("url_list")
    urls, classes = url_classifier.classify_urls(
        url_list, debug=True, return_predictions=True
    )

    return_value = {"urls_list": urls, "classes": classes}

    return jsonify(return_value)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
