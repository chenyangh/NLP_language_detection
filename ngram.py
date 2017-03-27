from os import listdir
from os.path import isfile, join
import string
from nltk.util import ngrams
from math import log, exp

voc_size = {}
smoothing = "Laplace"

def read_files(dir_path, is_using_space=True):
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    global voc_size
    data = {}
    for file in files:
        lang = file.split('.')[0].split('-')[1]
        voc = set()
        with open(join(dir_path, file)) as f:
            text = f.read()
            # Remove  punctuation TODO: remove Space?
            if is_using_space:
                data[lang] = ' '.join(filter(None,
                                     (word.strip(string.punctuation) for word in text.split())))
            else:
                data[lang] = ''.join(filter(None,
                                    (word.strip(string.punctuation) for word in text.split())))
            for letter in data[lang]:
                if letter not in voc and letter != ' ':
                    voc.add(letter)
            voc_size[lang] = len(voc)
    return data, voc_size


def get_grams(data, n_of_grams, is_padding=False, is_test=False):
    grams_dict = {}
    all_grams = []
    for i in n_of_grams:
        for word in data.split():
            generated_ngrams = ngrams(word, i, pad_left=is_padding, pad_right=False)
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
def grams_tf_idf_score(train_lang, test_grams, target_grams):
    total_target_grams = sum(target_grams.values())
    score = 0
    for test_gram in test_grams:
        if test_gram in target_grams:
            score += log(total_target_grams/target_grams[test_gram])
    return score


def language_model_score(train_lang, test_grams, target_grams):
    global voc_size
    a = voc_size

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
    if smoothing == "None":
        for test_gram in test_grams:
            if test_gram in target_grams:
                score += log(1/(target_grams[test_gram]/n_minus_one_gram_dict[test_gram[:n]]))
            else:
                return 0
    elif smoothing == "Laplace":
        for test_gram in test_grams:
            if test_gram in target_grams:
                cn = target_grams[test_gram]
                cn_minus_one = n_minus_one_gram_dict[test_gram[:n]]
            else:
                cn = 0
                if test_gram[:n] in n_minus_one_gram_dict:
                    cn_minus_one = n_minus_one_gram_dict[test_gram[:n]]
                else:
                    cn_minus_one = 0
            score += log(1/(
                            (cn+1) /
                            (cn_minus_one+voc_size[train_lang])
                            ))
    # print(score)

    return exp(score/total_test_grams)


def get_result(train_grams, dev_grams, score_method):
    correct = 0
    for test_lang in dev_grams:
        lang1 = test_lang
        lang2 = ''
        min_val = 999999999

        for target_lang in train_grams:

            val = score_method(target_lang, dev_grams[test_lang], train_grams[target_lang])
            if val < min_val:
                min_val = val
                lang2 = target_lang
        total_target_grams = len(train_grams[test_lang])
        if lang2 == lang1:
            correct += 1
        print(lang1, lang2, min_val, str(total_target_grams))
    print(correct / len(dev_grams))


def __main__():
    # Settings
    n_of_grams = [3]
    if_padding = True
    score_method = language_model_score
    train_path = '650_a3_train'
    dev_path = '650_a3_dev'

    # Read the file
    train_data, voc_size = read_files(train_path)
    dev_data, _ = read_files(dev_path)
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
