from .classifier import UrlClassifier

url_classifier = UrlClassifier()


url_list = [
    "https://developers.google.com/machine-learning/crash-course",
    "https://www.tensorflow.org/",
    "https://en.wikipedia.org/wiki/Algorithm#Expressing_algorithms",
    "https://deltacouncil.ca.gov/social-science",
    "https://www.investopedia.com/terms/e/economy.asp",
]


print(url_classifier.classify_urls(url_list))
