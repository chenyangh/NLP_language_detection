from os import listdir
from os.path import isfile, join
import string
from nltk.util import ngrams

def read_files(dir_path):
    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    data = {}
    for file in onlyfiles:
        lang = file.split('.')[0].split('-')[1]
        with open(join(dir_path, file)) as f:
            text = f.read()
            # Remove  punctuation TODO: remove Space?
            data[lang] = ' '.join(filter(None,
                                        (word.strip(string.punctuation) for word in text.split())))
    return data


def word_grams(words, min=1, max=4):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s


def __main__():
    train_path = '650_a3_train'
    dev_path = '650_a3_dev'
    train_data = read_files(train_path)
    dev_data = read_files(dev_path)
    test_data = train_data['als']
    generated_ngrams = ngrams(test_data, 4, pad_left=True, pad_right=True)



__main__()
