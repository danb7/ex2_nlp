from collections import Counter
import random
import gensim.downloader as dl
from transformers import RobertaTokenizerFast, RobertaForMaskedLM
import torch
import time
import utils

random.seed(42)

class Base_POS_Model():
    def fit(self, train_data):
        self.data_sentences = train_data
        self.train_word_pos_list = self.get_all_words_pos(self.data_sentences)
        self.train_word_pos_freq = self.get_word_freq_per_pos()
        self.pos_freq = self.get_pos_freq()
        self.word_most_freq_pos = self.get_word_most_freq_pos()
        self.most_freq_pos = self.get_most_freq_pos()

    def get_all_words_pos(sef, data):
        pos_list = []
        for sen in data:
            for token in sen.split():
                word_split = token.rsplit('/', 1)
                word_pos = (word_split[0], word_split[1])
                pos_list.append(word_pos)
        
        return pos_list

    def get_word_freq_per_pos(self):
        counts = {}
        for word, pos in self.train_word_pos_list:
            if word not in counts:
                counts[word] = {}

            counts[word][pos] = counts[word].get(pos, 0) + 1
            
        return counts

    def get_word_most_freq_pos(self):
        return {word: max(self.train_word_pos_freq[word], key=self.train_word_pos_freq[word].get) for word in self.train_word_pos_freq}
    
    def get_pos_freq(self):
        return Counter([pos for word, pos in self.train_word_pos_list])
    
    def get_most_freq_pos(self):
        all_pos_tags = [pos_tag for pos_dict in self.train_word_pos_freq.values() for pos_tag in pos_dict]

        return max(set(all_pos_tags), key=all_pos_tags.count)

    def predict(self, test_data, params):
        random.seed(42)
        word_pos_pred = []
        for sentence in test_data:
            for word in sentence.split():
                if word not in self.train_word_pos_freq:
                    if params['fill_sample']:
                        pos = random.choices(list(self.pos_freq.keys()), weights = list(self.pos_freq.values()))[0]
                    else:
                        pos = self.most_freq_pos

                    word_pos_pred.append(pos)
                else:
                    if params['word_sample']:
                        pos = random.choices(list(self.train_word_pos_freq[word].keys()), weights = list(self.train_word_pos_freq[word].values()))[0]
                    else:
                        pos = self.word_most_freq_pos[word]

                    word_pos_pred.append(pos)
        
        return word_pos_pred
    
    def calc_Accuracy(self, y_pred, y_true):
        if len(y_pred) != len(y_true):
            raise ValueError("Lists need to be the same length")
        
        return sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)

    def save_prediction_to_file(self, test_data, y_pred, test_file_path):
        output_string_list = []
        i = 0
        for sentence in test_data:
            sentence_tag = []
            for word in sentence.split():
                sentence_tag.append(f'{word}/{y_pred[i]}')
                i += 1

            output_string_list.append(f"{' '.join(sentence_tag)}\n")

        with open(test_file_path, 'w') as file:
            file.writelines(output_string_list)            

class Bigram_POS_Model(Base_POS_Model):
    def fit(self, train_data):
        super().fit(train_data)
        self.train_pos_word_most_freq_pos = self.get_bigram_dict()

    def get_bigram_dict(self):
        counts = {}
        for sentence in self.data_sentences:
            prev_pos = ''
            for token in sentence.split():
                word_split = token.rsplit('/', 1)
                pos_word = f'{prev_pos}_{word_split[0]}'
                if pos_word not in counts:
                    counts[pos_word] = {}

                counts[pos_word][word_split[1]] = counts[pos_word].get(word_split[1], 0) + 1
                prev_pos = word_split[1]
                
        return {word: max(counts[word], key=counts[word].get) for word in counts}

    def bigram_predict(self, test_data):
        random.seed(42)
        word_pos_pred = [] 
        for sentence in test_data:
            prev_pos = ''
            for word in sentence.split():
                pos = self.single_predict(prev_pos, word)
                word_pos_pred.append(pos)
                prev_pos = pos
        
        return word_pos_pred
    
    def single_predict(self, prev_pos, word):
        if f'{prev_pos}_{word}' not in self.train_pos_word_most_freq_pos:
            if word not in self.train_word_pos_freq:
                pos = self.most_freq_pos
            else:
                pos = self.word_most_freq_pos[word]
        else:
            pos = self.train_pos_word_most_freq_pos[f'{prev_pos}_{word}']

        return pos

    def bigram_predict_with_inflection(self, test_data):
        word_pos_pred = []          
        for sentence in test_data:
            prev_pos = ''
            for word in sentence.split():
                if all(w not in self.train_pos_word_most_freq_pos for w in [f'{prev_pos}_{word}',
                                                            f'{prev_pos}_{word.lower()}',
                                                            f'{prev_pos}_{word.capitalize()}']):
                    if all(w not in self.train_word_pos_freq for w in [word, word.lower(), word.capitalize()]):
                        pos = self.most_freq_pos
                    else:
                        if word in self.train_word_pos_freq:
                            pass
                        elif word.lower() in self.train_word_pos_freq:
                            word = word.lower()
                        elif word.capitalize() in self.train_word_pos_freq:
                            word = word.capitalize()
                        pos = self.word_most_freq_pos[word]
                else:
                    if f'{prev_pos}_{word}' in self.train_pos_word_most_freq_pos:
                        pass
                    elif f'{prev_pos}_{word.lower()}' in self.train_pos_word_most_freq_pos:
                        word = word.lower()
                    elif f'{prev_pos}_{word.capitalize()}' in self.train_pos_word_most_freq_pos:
                        word = word.capitalize()
                    pos = self.train_pos_word_most_freq_pos[f'{prev_pos}_{word}']

                word_pos_pred.append(pos)
                prev_pos = pos
        
        return word_pos_pred

class Static_Vector_POS_Model(Bigram_POS_Model):
    def fit(self, train_data):
        super().fit(train_data)
        self.similarity_model = dl.load("glove-wiki-gigaword-300") # word2vec-google-news-300 glove-twitter-200 the best

    def get_top_k_known_similar_words(self, word, k = 5):
        '''ensuring known words'''
        k_counter = 0
        similar_known_words = []
        try:
            most_similar_words = self.similarity_model.most_similar(word, topn=100)
            for similar_word, cos_similar in most_similar_words:
                if (similar_word in self.train_word_pos_freq) and (similar_word.isalpha()) and (len(similar_word)>1):
                    k_counter += 1
                    similar_known_words.append(similar_word)
                if k_counter == k:
                    break
        except Exception:
            pass 
        
        return similar_known_words

    def bigram_predict_static(self, test_data, k = 5):
        word_pos_pred = []
        for sentence in test_data:
            prev_pos = ''
            for word in sentence.split():
                pos = None
                if f'{prev_pos}_{word}' not in self.train_pos_word_most_freq_pos:
                    if word not in self.train_word_pos_freq:
                        similar_word = self.get_top_k_known_similar_words(word, k)
                        possible_pos_fills = [self.train_pos_word_most_freq_pos.get(f'{prev_pos}_{w}', self.word_most_freq_pos.get(w, self.most_freq_pos)) for w in similar_word]
                        possible_pos_fills = [self.most_freq_pos] if possible_pos_fills==[] else possible_pos_fills
                        fill_value = max(possible_pos_fills, key=possible_pos_fills.count)
                        #fill_value = Counter(possible_pos_fills).most_common(1)[0][0]
                        pos = fill_value
                    else:
                        pos = self.word_most_freq_pos[word]
                else:
                    pos = self.train_pos_word_most_freq_pos[f'{prev_pos}_{word}']

                word_pos_pred.append(pos)
                prev_pos = pos
        
        return word_pos_pred     
    
class Contextualized_Vector_POS_Model(Bigram_POS_Model):
    def fit(self, train_data, fill_model_name = 'roberta-base'):
        super().fit(train_data)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(fill_model_name)
        self.masked_language_model = RobertaForMaskedLM.from_pretrained(fill_model_name)

    def get_top_k_known_predictions(self, logits, word_index, tokenizer, k=5):
        '''ensuring known words'''
        probabilities = torch.nn.functional.softmax(logits[0, word_index, :], dim=-1)
        sort_indices = torch.argsort(probabilities, descending =True)
        k_counter = 0
        predicted_tokens = []
        predicted_probabilities = []
        for i in sort_indices:
            decoded_i = tokenizer.decode(i)
            decoded_i = decoded_i.strip()
            if (decoded_i in self.train_word_pos_freq) and (decoded_i.isalpha()) and (len(decoded_i)>1):
                k_counter += 1
                predicted_tokens += [decoded_i]
                predicted_probabilities += [probabilities[i].item()]
            if k_counter == k:
                break
            
        top_k_predictions = predicted_tokens[:k]
        top_k_probabilities = predicted_probabilities[:k]
        
        return {'predicted_tokens': top_k_predictions, 'probas': top_k_probabilities}
    
    def bigram_predict_with_contextualized(self, test_data, k=5):
        cum_fill_time = 0
        word_pos_pred = []
        start = time.time()
        for sentence in test_data:
            prev_pos = ''
            for i, word in enumerate(sentence.split()):
                if f'{prev_pos}_{word}' not in self.train_pos_word_most_freq_pos:
                    if word not in self.train_word_pos_freq:
                        # try first to get similar words before filling with mode
                        sentence_list = sentence.split()
                        sentence_list[i] = self.tokenizer.mask_token
                        masked_sentence = ' '.join(sentence_list)
                        tokenized_sentence = self.tokenizer(masked_sentence, return_tensors='pt')
                        mask_index = tokenized_sentence["input_ids"][0].tolist().index(self.tokenizer.mask_token_id)
                        with torch.no_grad():
                            logits = self.masked_language_model(**tokenized_sentence).logits
                        start_fill = time.time()
                        #top_k_words = get_top_k_known_predictions2(logits, mask_index, tokenizer, known_vocab, k)['predicted_tokens']
                        top_k_words = self.get_top_k_known_predictions(logits, mask_index, self.tokenizer, k)['predicted_tokens']
                        cum_fill_time += (time.time()-start_fill)
                        # print(cum_fill_time)
                        possible_pos_fills = [self.train_pos_word_most_freq_pos.get(f'{prev_pos}_{w}', self.word_most_freq_pos.get(w, self.most_freq_pos)) for w in top_k_words]
                        possible_pos_fills = [self.most_freq_pos] if possible_pos_fills==[] else possible_pos_fills
                        fill_value = max(possible_pos_fills, key=possible_pos_fills.count)
                        pos = fill_value
                    else:
                        pos = self.word_most_freq_pos[word]
                else:
                    pos = self.train_pos_word_most_freq_pos[f'{prev_pos}_{word}']

                word_pos_pred.append(pos)
                prev_pos = pos

        print(time.time()-start, cum_fill_time, time.time()-start-cum_fill_time, )
        return word_pos_pred
