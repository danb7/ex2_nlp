import codecs
from collections import Counter
import random

def get_file_data(file_name, lower = False, as_lists=False):
    with open(file_name, 'r') as file:   
        file_data = file.read().splitlines()
    
    if lower:
        file_data = [s.lower() for s in file_data]
    
    if as_lists:
        file_data = [sentence.split() for sentence in file_data]
    return file_data

def get_all_words_no_anotation(data):
    words = []
    for sen in data:
        for word in sen.split(' '):
            words.append(word)
    
    return words

def get_all_words_pos(data):
    pos_list = []
    for sen in data:
        for token in sen.split():
            word_split = token.rsplit('/', 1)
            word_pos = (word_split[0], word_split[1])
            pos_list.append(word_pos)
    
    return pos_list

def get_word_freq_per_pos(data):
    counts = {}
    for word, pos in data:
        if word not in counts:
            counts[word] = {}

        counts[word][pos] = counts[word].get(pos, 0) + 1
        
    return counts

def get_word_most_freq_pos(data):
    return {word: max(data[word], key=data[word].get) for word in data}

def fill_for_missing_word(data):
    all_pos_tags = [pos_tag for pos_dict in data.values() for pos_tag in pos_dict]

    return max(set(all_pos_tags), key=all_pos_tags.count)

def predict(train_pos_dist_data, test_data, fill_pos_dist, word_sample = False, fill_sample = False):
    random.seed(42)
    word_pos_pred = []
    if not word_sample:
        train_word_most_freq_pos = get_word_most_freq_pos(train_pos_dist_data)
    if not fill_sample:
        fill_value = fill_for_missing_word(train_pos_dist_data)
        
    for sentence in test_data:
        for word in sentence.split():
            if word not in train_pos_dist_data:
                if fill_sample:
                    pos = random.choices(list(fill_pos_dist.keys()), weights = list(fill_pos_dist.values()))[0]
                else:
                    pos = fill_value

                word_pos_pred.append(pos)
            else:
                if word_sample:
                    pos = random.choices(list(train_pos_dist_data[word].keys()), weights = list(train_pos_dist_data[word].values()))[0]
                else:
                    pos = train_word_most_freq_pos[word]

                word_pos_pred.append(pos)
    
    return word_pos_pred

def calc_Accuracy(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError("Lists need to be the same length")
    
    return sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)

def predict_with_inflection(train_pos_dist_data, test_data, fill_pos_dist, word_sample = False, fill_sample = False):
    random.seed(42)
    word_pos_pred = []
    if not word_sample:
        train_word_most_freq_pos = get_word_most_freq_pos(train_pos_dist_data)
    if not fill_sample:
        fill_value = fill_for_missing_word(train_pos_dist_data)
        
    for sentence in test_data:
        for word in sentence.split():
            if all(w not in train_pos_dist_data for w in [word, word.lower(), word.capitalize()]):
                if fill_sample:
                    pos = random.choices(list(fill_pos_dist.keys()), weights = list(fill_pos_dist.values()))[0]
                else:
                    pos = fill_value

                word_pos_pred.append(pos)
            else:
                if word in train_pos_dist_data:
                    pass
                elif word.lower() in train_pos_dist_data:
                    word = word.lower()
                elif word.capitalize() in train_pos_dist_data:
                    word = word.capitalize()
                    
                if word_sample:
                    pos = random.choices(list(train_pos_dist_data[word].keys()), weights = list(train_pos_dist_data[word].values()))[0]
                else:
                    pos = train_word_most_freq_pos[word]

                word_pos_pred.append(pos)
    
    return word_pos_pred

def get_bigram_dict(data):
    counts = {}
    for sentence in data:
        prev_pos = ''
        for token in sentence.split():
            word_split = token.rsplit('/', 1)
            pos_word = f'{prev_pos}_{word_split[0]}'
            if pos_word not in counts:
                counts[pos_word] = {}

            counts[pos_word][word_split[1]] = counts[pos_word].get(word_split[1], 0) + 1
            prev_pos = word_split[1]
            
    return counts

def bigram_predict(train_pos_word_dist_data, train_pos_dist_data, test_data, fill_pos_dist, word_sample = False, fill_sample = False):
    random.seed(42)
    word_pos_pred = []
    train_pos_word_most_freq_pos = get_word_most_freq_pos(train_pos_word_dist_data)
    if not word_sample:
        train_word_most_freq_pos = get_word_most_freq_pos(train_pos_dist_data)
    if not fill_sample:
        fill_value = fill_for_missing_word(train_pos_dist_data)
        
    for sentence in test_data:
        prev_pos = ''
        for word in sentence.split():
            if f'{prev_pos}_{word}' not in train_pos_word_dist_data:
                if word not in train_pos_dist_data:
                    pos = fill_value
                else:
                    pos = train_word_most_freq_pos[word]
            else:
                pos = train_pos_word_most_freq_pos[f'{prev_pos}_{word}']

            word_pos_pred.append(pos)
            prev_pos = pos
    
    return word_pos_pred

def bigram_predict_with_inflection(train_pos_word_dist_data, train_pos_dist_data, test_data, fill_pos_dist, word_sample = False, fill_sample = False):
    random.seed(42)
    word_pos_pred = []
    train_pos_word_most_freq_pos = get_word_most_freq_pos(train_pos_word_dist_data)
    if not word_sample:
        train_word_most_freq_pos = get_word_most_freq_pos(train_pos_dist_data)
    if not fill_sample:
        fill_value = fill_for_missing_word(train_pos_dist_data)
        
    for sentence in test_data:
        prev_pos = ''
        for word in sentence.split():
            if all(w not in train_pos_word_dist_data for w in [f'{prev_pos}_{word}',
                                                          f'{prev_pos}_{word.lower()}',
                                                          f'{prev_pos}_{word.capitalize()}']):
                if all(w not in train_pos_dist_data for w in [word, word.lower(), word.capitalize()]):
                    pos = fill_value
                else:
                    if word in train_pos_dist_data:
                        pass
                    elif word.lower() in train_pos_dist_data:
                        word = word.lower()
                    elif word.capitalize() in train_pos_dist_data:
                        word = word.capitalize()
                    pos = train_word_most_freq_pos[word]
            else:
                if f'{prev_pos}_{word}' in train_pos_word_dist_data:
                    pass
                elif f'{prev_pos}_{word.lower()}' in train_pos_word_dist_data:
                    word = word.lower()
                elif f'{prev_pos}_{word.capitalize()}' in train_pos_word_dist_data:
                    word = word.capitalize()
                pos = train_pos_word_most_freq_pos[f'{prev_pos}_{word}']

            word_pos_pred.append(pos)
            prev_pos = pos
    
    return word_pos_pred

##################### For NER #####################
def _read_data(fname):
    '''from ner_eval.py'''
    for line in codecs.open(fname):
        line = line.strip().split()
        tagged = [x.rsplit("/",1) for x in line]
        yield tagged

def _normalize_bio(tagged_sent):
    '''from ner_eval.py'''
    last_bio, last_type = "O","O"
    normalized = []
    for word, tag in tagged_sent:
        if tag == "O": tag = "O-O"
        bio,typ = tag.split("-",1)
        if bio=="I" and last_bio=="O": bio="B"
        if bio=="I" and last_type!=typ: bio="B"
        normalized.append((word,(bio,typ)))
        last_bio,last_type=bio,typ
    return normalized

def _get_entities(sent):
    ent=[]
    for i,(word,tag) in enumerate(sent):
        bio,typ=tag
        if bio=="B":
            if ent: yield tuple(ent)
            ent=[]
            ent.append(i)
            ent.append(typ)
            ent.append(word)
        if bio=="I":
            ent.append(word)
        if bio=="O":
            if ent: yield tuple(ent)
            ent=[]
    if ent: yield tuple(ent)
    
def get_ner_data(file_name):
    return [_normalize_bio(tagged) for tagged in _read_data(file_name)]

def get_ner_data_with_no_tags(file_name):
    data = []
    for tagged_sentence in _read_data(file_name):
        sent = [word for word, tag in tagged_sentence]
        data += [sent]
    return data