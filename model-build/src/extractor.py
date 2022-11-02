import pandas as pd
import concurrent
from IPython.display import display
import time

from .utils import get_text_p


class TextExtractor:
    def __init__(self, thread_count=10):
        self.num_threads = thread_count

    def extract(self, url_list, debug=False):
        df_dict = {}
        i = 0

        print(url_list)

        total_urls = len(url_list)

        # maybe some network error or something
        extracted = False

        while not extracted:
            try:
                # for each row in the data
                start = 0
                increment = self.num_threads

                while start < total_urls:
                    if debug:
                        print("Submitting threads...")
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.num_threads
                    ) as executor:
                        # get the urls
                        url_range = url_list[start : start + increment]

                        for url in url_range:
                            executor.submit(get_text_p, url, i, df_dict)
                            i += 1
                    if debug:
                        print("All threads completed!")
                        print(f"Completed {start} to {start + increment}...")

                    # increment the starting index
                    start += increment

                extracted = True
            except:
                time.sleep(5)

        url_dataframe = pd.DataFrame.from_dict(
            df_dict, columns=["url", "text"], orient="index"
        )
        url_dataframe.reset_index(inplace=True)
        try:
            url_dataframe.drop(["index"], axis=1, inplace=True)
        except:
            pass

        if debug:
            display(url_dataframe.head(n=5))

        return url_dataframe
