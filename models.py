from collections import Counter
import random
import re
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

    def get_all_words_pos(self, data):
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

class Base_NER_Model():

    def __init__(self, extension={}):
        '''
        extension_dict need to be a dict with the format:
        {
            'language_model': LM,
            'tokenizer': Tokenizer
            'topk_words': Number, default 50
            'topk_sims': Number, default 3
        }
        '''
        self.extension_dict = extension
        self.language_model = extension.get('language_model', None)
        self.tokenizer = extension.get('tokenizer', None)
        self.topk_words = extension.get('topk_words', 50)
        self.topk_sims = extension.get('topk_sims', 3)

    def fit(self, train_data):
        self.data_sentences = train_data
        self.train_word_ner_freq = self._get_word_freq_per_ner()
        self.word_most_freq_tag = self._get_word_most_freq_tag()
        self.most_freq_tag = self._get_most_freq_tag()
        if self.extension_dict:
            self.word_most_freq_tag.update(self._extend_data(self.topk_words, self.topk_sims))

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
    

    def get_top_k_oov_predictions(self, logits, word_index, k=5):#######################
        '''ensuring known words'''
        probabilities = torch.nn.functional.softmax(logits[0, word_index, :], dim=-1)
        sort_indices = torch.argsort(probabilities, descending =True)
        k_counter = 0
        predicted_tokens = []
        predicted_probabilities = []
        for i in sort_indices:
            decoded_i = self.tokenizer.decode(i)
            decoded_i = decoded_i.strip()
            if (decoded_i not in self.train_word_ner_freq) and (decoded_i.isalpha()) and (len(decoded_i)>1):
                k_counter += 1
                predicted_tokens += [decoded_i]
                predicted_probabilities += [probabilities[i].item()]
            if k_counter == k:
                break
            
        top_k_predictions = predicted_tokens[:k]
        top_k_probabilities = predicted_probabilities[:k]
        
        return {'predicted_tokens': top_k_predictions, 'probas': top_k_probabilities}
    
    def _get_first_occur_sentence_of_word_and_tag(self, target_word, target_tag):
        '''return a sentence contain this word with this tag and its position'''
        for sentence in self.data_sentences:
            for i, (word, (bio, ner_tag)) in enumerate(sentence):
                if (word==target_word) and (ner_tag==target_tag):
                    plain_sentence = [word2 for (word2, (bio2, ner_tag2)) in sentence]
                    return plain_sentence, i
        raise ValueError("word not in data")
    
    def _extend_data(self, topk_words=50, topk_sims=3):#######################
        extension = {}
        sorted_words_in_train = dict(sorted(self.train_word_ner_freq.items(), key=lambda x: sum(x[1].values()), reverse=True))
        filtered_sorted_words_in_train = {key: value for key, value in sorted_words_in_train.items() if max(value.values()) != value.get('O', 0)}
        top_words = list(filtered_sorted_words_in_train.keys())[:topk_words]
        for word in top_words:
            word_most_frq_tag = self.word_most_freq_tag[word]
            sent, loc = self._get_first_occur_sentence_of_word_and_tag(word, word_most_frq_tag)           
            masked_sentence = ' '.join([word if i != loc else '<mask>' for i, word in enumerate(sent)])

            tokenized_sentence = self.tokenizer(masked_sentence, return_tensors='pt')
            mask_index = tokenized_sentence["input_ids"][0].tolist().index(self.tokenizer.mask_token_id)
            with torch.no_grad():
                logits = self.language_model(**tokenized_sentence).logits
            top_k_words = self.get_top_k_oov_predictions(logits, mask_index, topk_sims)['predicted_tokens']
            for w in top_k_words:
                extension[w] = word_most_frq_tag
        return extension

    def _inflect_word_if_missing(self, word):
        if word not in self.train_word_ner_freq:
            if word.capitalize() in self.train_word_ner_freq: word = word.capitalize()
            if word.lower() in self.train_word_ner_freq: word = word.lower()
        return word


    def predict(self, test_data, inflect_missing=False):
        words_preds = []
        for sentence in test_data:
            for word in sentence:
                if inflect_missing:
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

class NER_Model_With_PCA_Feats(Base_NER_Model):
    '''
    Expand our "features" by add to each word its "features" based on PCA components
    '''
    
    def __init__(self, fitted_pos_model, include_previous=False):
        self.pretrained_pos_model = fitted_pos_model
        self.include_previous = include_previous

    def fit(self, train_data):
        super().fit(train_data)
        self.data_expansion = self._add_word_pos_to_data()
        self.word_pos_ner_freq = self._get_ner_freq_per()
        self.most_freq_tag_per = self._get_feats_most_freq_tag
        
    def _extract_sentences(self, data):
        plain_sentences = []
        for sentence in data:
            sentence_list = [word for (word, (bio, ner_tag)) in sentence]
            plain_sentences += [' '.join(sentence_list)]
        return plain_sentences

    def _add_word_pos_to_data(self):
        expanded_data = []
        pos_model_input = self._extract_sentences(self.data_sentences)
        pos_tags = self.pretrained_pos_model.bigram_predict_with_contextualized(pos_model_input) # need to be all predict
        i=0
        for sentence in self.data_sentences:
            sentence_list = []
            for w_loc, (word, (bio, ner_tag)) in enumerate(sentence):
                if self.include_previous:
                    prev_pos = pos_tags[i-1] if w_loc!=0 else ''
                    sentence_list += [(word, (bio, ner_tag, pos_tags[i], prev_pos))] # pay attention to previous
                else:
                    sentence_list += [(word, (bio, ner_tag, pos_tags[i]))]
                i+=1
            expanded_data += sentence_list
        return expanded_data

    def _get_ner_freq_per(self):
        counts = {}
        for sentence in self.data_expansion:
            if self.include_previous:
                for (word, (bio, ner_tag, pos_tag)) in sentence:
                    feats = f'{word}_{pos_tag}'
                    if feats not in counts: # necessary?
                        counts[feats] = {}
                    counts[feats][ner_tag] = counts[feats].get(ner_tag, 0) + 1
            else:
                for (word, (bio, ner_tag, pos_tag, prev_pos_tag)) in sentence:
                    feats = f'{word}_{pos_tag}_{prev_pos_tag}'
                    if feats not in counts: # necessary?
                        counts[feats] = {}
                    counts[feats][ner_tag] = counts[feats].get(ner_tag, 0) + 1          
        return counts
    
    def _get_feats_most_freq_tag(self):
        return {feats: max(self.freq[feats], key=self.freq[feats].get) for feats in self.freq}

    def predict(self, test_data, inflect_missing=False):
        pass


class NER_Model_With_POS_Feats(Base_NER_Model):
    '''
    Add to our features the POS tag of words with our best POS model
    '''
    
    def __init__(self, fitted_pos_model, model_dict):
        '''
        model_dict need to be a dict in the following format:
        {
            'language_model': LM,
            'tokenizer': Tokenizer
        }
        '''
        super().__init__()
        self.pretrained_pos_model = fitted_pos_model
        self.language_model = model_dict.get('language_model', None)
        self.tokenizer = model_dict.get('tokenizer', None)
        self.test_counter = 0
        self.capitalized_dict = {}

    def fit(self, train_data):
        super().fit(train_data)
        self.data_expansion = self._add_word_pos_to_data(include_previous=False)
        self.word_pos_ner_freq = self._get_ner_freq_per(self.data_expansion, include_previous=False)
        self.most_freq_tag_per = self._get_feats_most_freq_tag(include_previous=False)

        self.data_expansion_with_prev = self._add_word_pos_to_data(include_previous=True)
        self.word_pos_ner_freq_with_prev = self._get_ner_freq_per(self.data_expansion_with_prev, include_previous=True)
        self.most_freq_tag_per_with_prev = self._get_feats_most_freq_tag(include_previous=True)

        # self.most_freq_tag_by_prev_pos = self._get_most_freq_tag_by_prev_pos()
        self.ner_freq_tag_by_pos = self._get_most_freq_tag_by_pos(arg_max=False)
        self.most_freq_tag_by_pos = self._get_most_freq_tag_by_pos()
        self.most_freq_tag_by_pos_w_prev = self._get_most_freq_tag_by_pos(include_previous=True)
        
    def _extract_sentences(self, data, is_test=False):
        plain_sentences = []
        for sentence in data:
            if not is_test:
                sentence_list = [word for (word, (bio, ner_tag)) in sentence]
            else:
                sentence_list = sentence
            plain_sentences += [' '.join(sentence_list)]
        return plain_sentences

    def _add_word_pos_to_data(self, include_previous=False):
        expanded_data = []
        pos_model_input = self._extract_sentences(self.data_sentences)
        pos_tags = self.pretrained_pos_model.bigram_predict(pos_model_input) # need to be all predict
        i=0
        for sentence in self.data_sentences:
            sentence_list = []
            for w_loc, (word, (bio, ner_tag)) in enumerate(sentence):
                if include_previous:
                    prev_pos = pos_tags[i-1] if w_loc!=0 else ''
                    sentence_list += [(word, (bio, ner_tag, pos_tags[i], prev_pos))] # pay attention to previous
                else:
                    sentence_list += [(word, (bio, ner_tag, pos_tags[i]))]
                i+=1
            expanded_data += [sentence_list]
        return expanded_data

    def _get_ner_freq_per(self, data_exp, include_previous=False):
        counts = {}
        for sentence in data_exp:
            if include_previous:
                for (word, (bio, ner_tag, pos_tag, prev_pos_tag)) in sentence:
                    feats = f'{word}_{pos_tag}_{prev_pos_tag}'
                    if feats not in counts:
                        counts[feats] = {}
                    counts[feats][ner_tag] = counts[feats].get(ner_tag, 0) + 1
            else:
                for (word, (bio, ner_tag, pos_tag)) in sentence:
                    feats = f'{word}_{pos_tag}'
                    if feats not in counts:
                        counts[feats] = {}
                    counts[feats][ner_tag] = counts[feats].get(ner_tag, 0) + 1  
        return counts
    
    def _get_feats_most_freq_tag(self, include_previous=False):
        if include_previous:
            result = {feats: max(self.word_pos_ner_freq_with_prev[feats], key=self.word_pos_ner_freq_with_prev[feats].get) for feats in self.word_pos_ner_freq_with_prev}
        else:
            result = {feats: max(self.word_pos_ner_freq[feats], key=self.word_pos_ner_freq[feats].get) for feats in self.word_pos_ner_freq}
        return result
    
    def _get_most_freq_tag_by_prev_pos(self):
        tags_by_prev_pos_counts = {}
        for sentence in self.data_expansion_with_prev:
                for (word, (bio, ner_tag, pos_tag, prev_pos_tag)) in sentence:
                    if prev_pos_tag not in tags_by_prev_pos_counts:
                        tags_by_prev_pos_counts[prev_pos_tag] = {}
                    tags_by_prev_pos_counts[prev_pos_tag][ner_tag] = tags_by_prev_pos_counts[prev_pos_tag].get(ner_tag, 0) + 1
        
        max_tags_by_prev_pos_counts = {prev: max(tags_by_prev_pos_counts[prev], key=tags_by_prev_pos_counts[prev].get) for prev in tags_by_prev_pos_counts}
        return max_tags_by_prev_pos_counts
    
    def _get_most_freq_tag_by_pos(self, include_previous=False, arg_max=True):
        tags_by_pos_counts = {}
        for sentence in self.data_expansion_with_prev:
                for (word, (bio, ner_tag, pos_tag, prev_pos_tag)) in sentence:
                    pos_feat = f'{prev_pos_tag}_{pos_tag}' if include_previous else pos_tag
                    if pos_feat not in tags_by_pos_counts:
                        tags_by_pos_counts[pos_feat] = {}
                    tags_by_pos_counts[pos_feat][ner_tag] = tags_by_pos_counts[pos_feat].get(ner_tag, 0) + 1
        
        max_tags_by_pos_counts = {feat: max(tags_by_pos_counts[feat], key=tags_by_pos_counts[feat].get) for feat in tags_by_pos_counts}
        return_dict = max_tags_by_pos_counts if arg_max else tags_by_pos_counts
        return return_dict
    
    def _get_top_k_known_predictions(self, logits, word_index, k=5):
        '''ensuring known words'''
        sort_indices = torch.argsort(logits[0, word_index, :], descending =True)
        k_counter = 0
        predicted_tokens = []
        for i in sort_indices:
            decoded_i = self.tokenizer.decode(i)
            decoded_i = decoded_i.strip()
            if (decoded_i in self.word_most_freq_tag) and (decoded_i.isalpha()) and (len(decoded_i)>1):
                k_counter += 1
                predicted_tokens += [decoded_i]
            if k_counter == k:
                break
                    
        return predicted_tokens
    
    def _rule_based_fixing(self, word, original_ner, cur_pos, prev_pos, sentence, w_loc, inc_prev, prev_ner):
        if re.match(r'^[0-9-:]+$', word): # numbers
            return 'O'
        elif word in self.capitalized_dict:
            return self.capitalized_dict[word]
        elif (re.match(r'[A-Z][A-Za-z]+', word)) \
            and cur_pos in [k for k,v in self.ner_freq_tag_by_pos.items() if ((v['O']/sum(v.values()))<0.99) and (len(k)>1)and(k!="''")] \
            and sum(self.train_word_ner_freq.get(word,{}).values())<10 \
            and word not in ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
                             "Friday", "Saturday", "January", "February", "March", "April", "May",
                             "June", "July", "August", "September", "October", "November", "December",
                             "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                             "Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.",
                             "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
                             "The"] \
            and original_ner=='O' \
            and True:#w_loc != 0: # Capitalized words inside sentence that classified as 'O'
            masked_sentence = ' '.join([word if i != w_loc else '<mask>' for i, word in enumerate(sentence)])
            tokenized_sentence = self.tokenizer(masked_sentence, return_tensors='pt')
            mask_index = tokenized_sentence["input_ids"][0].tolist().index(self.tokenizer.mask_token_id)
            with torch.no_grad():
                logits = self.language_model(**tokenized_sentence).logits
            filling_options = self._get_top_k_known_predictions(logits, mask_index, k=3)
            possible_tags = [self._get_best_ner_tag(f, cur_pos, prev_pos, inc_prev) for f in filling_options]
            self.test_counter += 1
            if self.test_counter % 100 == 0:
                print(self.test_counter)
            chosen_tag = max(possible_tags, key=possible_tags.count)
            self.capitalized_dict[word] = chosen_tag
            return chosen_tag
        elif ((word in ('of', 'in')) and (prev_ner in ('ORG', 'MISC', 'LOC'))):
            return prev_ner
        else:
            return original_ner
    
    def _repassing_uncertain_entities(self, entity):
        tags = [tag for word, tag in entity]
        if (len(entity) > 1) and (len(set(tags))) > 1:
            if tags[-1] != 'PER':
                tags = tags[::-1]
            # mode_tag = max(set(tags), key=tags.count)
            mode_tag = max((tags), key=tags.count)
            entity = [(word, mode_tag) for word, tag in entity]
        return entity

    def _get_best_ner_tag(self, word, cur_pos, prev_pos, include_previous=True):
        best_ner_tag = self.most_freq_tag_per.get(f'{word}_{cur_pos}',
                                                     self.word_most_freq_tag.get(word,
                                                                                 self.most_freq_tag_by_pos_w_prev.get(f'{prev_pos}_{cur_pos}',
                                                                                                                      self.most_freq_tag_by_pos.get(cur_pos, 'O'))))
        if include_previous:
            best_ner_tag = self.most_freq_tag_per_with_prev.get(f'{word}_{cur_pos}_{prev_pos}', best_ner_tag)
        return best_ner_tag

    def predict(self, test_data, include_previous_pos=False, inflect_missing=False):
        words_preds = []
        pos_model_input = self._extract_sentences(test_data, is_test=True)
        pos_tags = self.pretrained_pos_model.bigram_predict(pos_model_input)
        pos_i = 0
        for sentence in test_data:
            prev_ner = ''
            ent = []
            for w_loc, word in enumerate(sentence):
                if inflect_missing:
                    word = self._inflect_word_if_missing(word)
                prev_pos = pos_tags[pos_i-1] if w_loc!=0 else ''
                ner_tag = self.most_freq_tag_per.get(f'{word}_{pos_tags[pos_i]}',
                                                     self.word_most_freq_tag.get(word,
                                                                                 self.most_freq_tag_by_pos_w_prev.get(f'{prev_pos}_{pos_tags[pos_i]}',
                                                                                                                      self.most_freq_tag_by_pos.get(pos_tags[pos_i], 'O'))))
                ner_tag = self._rule_based_fixing(word, ner_tag, pos_tags[pos_i], prev_pos, sentence, w_loc, include_previous_pos, prev_ner)
                if include_previous_pos:
                    ner_tag = self.most_freq_tag_per_with_prev.get(f'{word}_{pos_tags[pos_i]}_{prev_pos}', ner_tag)
                pos_i += 1
                prev_ner = ner_tag
                if ner_tag != 'O':
                    ent += [(word, ner_tag)]
                elif ent:
                    final_ent = self._repassing_uncertain_entities(ent)
                    fixed_ners = [t_n for word, t_n in final_ent]
                    modify_len = len(fixed_ners)
                    words_preds[-modify_len:] = fixed_ners
                    ent = []
                words_preds.append(ner_tag)

        return words_preds