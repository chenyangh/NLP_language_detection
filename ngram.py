from os import listdir
from os.path import isfile, join
import string
from nltk.util import ngrams
from math import log


def read_files(dir_path, is_using_space=True):
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    data = {}
    for file in files:
        lang = file.split('.')[0].split('-')[1]
        with open(join(dir_path, file)) as f:
            text = f.read()
            # Remove  punctuation TODO: remove Space?
            if is_using_space:
                data[lang] = ' '.join(filter(None,
                                     (word.strip(string.punctuation) for word in text.split())))
            else:
                data[lang] = ''.join(filter(None,
                                    (word.strip(string.punctuation) for word in text.split())))
    return data


def get_grams(data, n_of_grams, is_padding=False):
    grams_dict = {}
    for word in data.split():
        for i in n_of_grams:
            generated_ngrams = ngrams(word, i, pad_left=is_padding, pad_right=is_padding)
            grams = list(generated_ngrams)

            for gram in grams:
                if gram not in grams_dict:
                    grams_dict[gram] = 1
                else:
                    grams_dict[gram] += 1
    return grams_dict


# Get TF-IDF score of a language gram in a target gram:
def grams_tf_idf_score(test_grams, target_grams):
    total_target_grams = sum(target_grams.values())
    score = 0
    for test_gram in test_grams:
        if test_gram in target_grams:
            score += log(total_target_grams/target_grams[test_gram])
    return score


def __main__():
    train_path = '650_a3_train'
    dev_path = '650_a3_dev'
    train_data = read_files(train_path)
    dev_data = read_files(dev_path)
    # test_data = train_data['als']
    train_grams = {}
    n_of_grams = [2, 3, 4]
    for lang in train_data:
        grams_dict = get_grams(train_data[lang], n_of_grams, is_padding=False)
        train_grams[lang] = grams_dict

    dev_grams = {}
    for lang in dev_data:
        grams_dict = get_grams(dev_data[lang], n_of_grams, is_padding=False)
        dev_grams[lang] = grams_dict

    correct = 0
    for test_lang in dev_grams:
        lang1 = test_lang
        lang2 = ''
        max_val = -100

        for target_lang in train_grams:
            val = grams_tf_idf_score(dev_grams[test_lang], train_grams[target_lang])
            if val > max_val:
                max_val = val
                lang2 = target_lang

        if lang2 == lang1:
            correct += 1

    print(correct/len(dev_grams))

__main__()
