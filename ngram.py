from os import listdir
from os.path import isfile, join
import string
from nltk.util import ngrams
from math import log, pow


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


def get_grams(data, n_of_grams, is_padding=False, is_test=False):
    grams_dict = {}
    all_grams = []
    for i in n_of_grams:
        for word in data.split():
            generated_ngrams = ngrams(word, i, pad_left=is_padding, pad_right=is_padding)
            grams = list(generated_ngrams)
            all_grams.extend(grams)
            for gram in grams:
                if gram not in grams_dict:
                    grams_dict[gram] = 1
                else:
                    grams_dict[gram] += 1
    if is_test:
        return all_grams
    else:
        return grams_dict


# Get TF-IDF score of a language gram in a target gram:
def grams_tf_idf_score(test_grams, target_grams):
    total_target_grams = sum(target_grams.values())
    score = 0
    for test_gram in test_grams:
        if test_gram in target_grams:
            score += log(total_target_grams/target_grams[test_gram])
    return score


def language_model_score(test_grams, target_grams):
    total_test_grams = len(test_grams)
    # calculate the n-1 gram
    n = len(test_grams[0]) - 1
    n_minus_one_gram_dict = {}
    for gram in target_grams:
        n_minus_one_gram = gram[:n]
        if n_minus_one_gram not in n_minus_one_gram_dict:
            n_minus_one_gram_dict[n_minus_one_gram] = target_grams[gram]
        else:
            n_minus_one_gram_dict[n_minus_one_gram] += target_grams[gram]

    score = 0
    for test_gram in test_grams:
        if test_gram in target_grams:
            score += log(1/(target_grams[test_gram]/n_minus_one_gram_dict[test_gram[:n]]))
    # print(score)
    return score/total_test_grams


def get_result(train_grams, dev_grams, score_method):
    correct = 0
    for test_lang in dev_grams:
        lang1 = test_lang
        lang2 = ''
        max_val = -999999999

        for target_lang in train_grams:
            val = score_method(dev_grams[test_lang], train_grams[target_lang])
            if val > max_val:
                max_val = val
                lang2 = target_lang

        if lang2 == lang1:
            correct += 1

    print(correct / len(dev_grams))


def __main__():
    # Settings
    n_of_grams = [4]
    if_padding = False
    score_method = language_model_score
    train_path = '../650_a3_train'
    dev_path = '../650_a3_dev'

    # Read the file
    train_data = read_files(train_path)
    dev_data = read_files(dev_path)
    # Get grams
    train_grams = {}
    for lang in train_data:
        grams_dict = get_grams(train_data[lang], n_of_grams, is_padding=if_padding)
        train_grams[lang] = grams_dict

    dev_grams = {}
    for lang in dev_data:
        grams_dict = get_grams(dev_data[lang], n_of_grams, is_padding=if_padding, is_test=True)
        dev_grams[lang] = grams_dict

    # Get result
    print("The result of", score_method.__name__, "is:", end=' ')
    get_result(train_grams, dev_grams, score_method)


__main__()
