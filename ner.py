import codecs
import subprocess
import utils

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
    
def get_ner_data(file_name):
    return [_normalize_bio(tagged) for tagged in _read_data(file_name)]


def get_ner_data_with_no_tags(file_name):
    data = []
    for tagged_sentence in _read_data(file_name):
        sent = [word for word, tag in tagged_sentence]
        data += [sent]
    return data
        

class Base_NER_Model():

    def fit(self, train_data):
        self.data_sentences = train_data
        self.train_word_ner_freq = self._get_word_freq_per_ner()
        self.word_most_freq_tag = self._get_word_most_freq_tag()
        self.most_freq_tag = self._get_most_freq_tag()


    def _get_word_freq_per_ner(self):
        counts = {}
        for sentence in self.data_sentences:
            for (word, (bio, ner_tag)) in sentence:
                if word not in counts:
                    counts[word] = {}
                counts[word][ner_tag] = counts[word].get(ner_tag, 0) + 1
        return counts

    def _get_word_most_freq_tag(self):
        return {word: max(self.train_word_ner_freq[word], key=self.train_word_ner_freq[word].get) for word in self.train_word_ner_freq}
    
    def _get_most_freq_tag(self):
        all_tags = []
        for sentence in self.data_sentences:
            for (word, (bio, tag)) in sentence:
                all_tags += [tag]
        return max(set(all_tags), key=all_tags.count)
    
    def _inflect_word_if_missing(self, word):
        if word not in self.train_word_ner_freq:
            if word.capitalize() in self.train_word_ner_freq: word = word.capitalize()
            if word.lower() in self.train_word_ner_freq: word = word.lower()
        return word


    def predict(self, test_data):
        words_preds = []
        for sentence in test_data:
            for word in sentence:
                word = self._inflect_word_if_missing(word)
                if word not in self.train_word_ner_freq:
                    tag = self.most_freq_tag
                    words_preds.append(tag)
                else:
                    tag = self.word_most_freq_tag[word]
                    words_preds.append(tag)
        return words_preds
    

    def save_prediction_to_file(self, test_data, y_pred, test_file_path):
        output_string_list = []
        i = 0
        for sentence in test_data:
            sentence_tag = []
            for word in sentence:
                if y_pred[i]=='O':
                    tag = y_pred[i]
                else:
                    tag = f'I-{y_pred[i]}'
                sentence_tag.append(f'{word}/{tag}')
                i += 1

            output_string_list.append(f"{' '.join(sentence_tag)}\n")

        with open(test_file_path, 'w') as file:
            file.writelines(output_string_list)     



train_file = r'data\ner\train'
dev_file = r'data\ner\dev'

normalized_train_data = get_ner_data(train_file)
dev_sentences = get_ner_data_with_no_tags(dev_file)

ner_base_model = Base_NER_Model()
ner_base_model.fit(normalized_train_data)
y_dev_pred = ner_base_model.predict(dev_sentences)

ner_base_model.save_prediction_to_file(dev_sentences, y_dev_pred, r'ner_pred_on_dev_base_model.txt')
subprocess.run(['python', 'ner_eval.py', r'data\ner\dev', r'ner_pred_on_dev_base_model.txt'], check=True)