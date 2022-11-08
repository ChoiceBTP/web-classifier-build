import requests
import bs4
import pickle
import nltk
import pandas as pd

DATA_PATH = "database/stored_urls.csv"


def get_html(url):
    r = requests.get(url, headers={"User-Agent": "My User Agent 1.0"})
    return r.content


def get_text_p(url, id_, df_dict):
    html_content = get_html(url)
    soup = bs4.BeautifulSoup(html_content, features="html.parser")

    # strip all script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    url_text = soup.get_text(" ", strip=True)

    df_dict[id_] = [url, url_text]

    return


def remove_stop_words(text):
    word_tokens = nltk.tokenize.word_tokenize(text)

    # remove stopwords and special characters
    filtered = ""

    # populate stop words
    stop_words = nltk.corpus.stopwords.words("english")
    stop_words = set([s.lower() for s in stop_words])

    for w in word_tokens:
        first_cond = w not in stop_words

        # if all chars are not alnum, remove
        count = 0
        for c in w:
            if c.isalnum():
                break
            count += 1
        second_cond = not (count == len(w))

        if first_cond and second_cond:
            filtered += w + " "

    return filtered.strip()


def model_loader(model_type):
    # model type is in [vectorizer, ml-model]
    file_path = "models/" + model_type + ".pickle"
    with open(file_path, "rb") as pickle_file:
        model = pickle.load(pickle_file)
    return model


def present_in_db(list_urls):
    dataframe = pd.read_csv(DATA_PATH, low_memory=False)

    if dataframe.empty:
        return [], []

    stored_urls, classes = [], []

    # get all the urls of the database
    database = {
        url: class_label
        for url, class_label in zip(dataframe["url"], dataframe["class"])
    }
    print(database)
    # get the stored urls
    for url in list(list_urls):
        if url in database:
            stored_urls.append(url)
            classes.append(database[url])

    return stored_urls, classes


def store_classifications(urls, classes):
    # handle logic for starting and stuff

    # in any case, I just need to append the
    # classifications, I am assuming that the
    # database would be having the
    # correct column names in the csv file

    curr_dataframe = pd.read_csv(DATA_PATH, low_memory=False)

    # if we are just starting out
    if curr_dataframe.empty:
        curr_dataframe = pd.DataFrame(data=[], columns=["url", "class"])

    # make a dict of the urls, classes
    data_dict = {"url": urls, "class": classes}
    new_data = pd.DataFrame.from_dict(data=data_dict, orient="columns")

    # append the new data dict in to the dataframe
    new_dataframe = pd.concat([curr_dataframe, new_data])

    # store it as the new data
    new_dataframe.to_csv(DATA_PATH)

    return
