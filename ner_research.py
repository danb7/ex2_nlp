import re
import subprocess
from matplotlib import pyplot as plt

from sklearn import decomposition
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaForMaskedLM
import utils
import models


train_file = r'data\ner\train'
dev_file = r'data\ner\dev'

normalized_train_data = utils.get_ner_data(train_file)
dev_sentences = utils.get_ner_data_with_no_tags(dev_file)


# dev_w = [word for sentence in utils.get_ner_data(dev_file) for loc, (word, (bio,tag)) in enumerate(sentence) if (re.match(r'[A-Z][a-z]+', word)) and (loc!=0) and (tag=='O')]


train_words = set([word for sentence in normalized_train_data for (word, (bio,tag)) in sentence if tag!='O'])
dev_words = set([word for sentence in utils.get_ner_data(dev_file) for (word, (bio,tag)) in sentence if tag!='O'])
# dev_words = set([word for sentence in dev_sentences for word in sentence])

print("size of vocab i  dev and train: ", len(dev_words), len(train_words))
print("size of words that are in dev but not in train (OOV):", len(dev_words.difference(train_words)))
# #############################Base##############################
# ner_base_model = models.Base_NER_Model()
# ner_base_model.fit(normalized_train_data)

# y_dev_pred = ner_base_model.predict(dev_sentences)          
# ner_base_model.save_prediction_to_file(dev_sentences, y_dev_pred, r'ner_pred_on_dev_base_model.txt')
# print('basic NER predictions')
# subprocess.run(['python', 'ner_eval.py', r'data\ner\dev', r'ner_pred_on_dev_base_model.txt'], check=True)

# y_dev_pred = ner_base_model.predict(dev_sentences, inflect_missing=True)
# ner_base_model.save_prediction_to_file(dev_sentences, y_dev_pred, r'ner_pred_on_dev_base_model.txt')
# print('basic NER predictions. inflect missing words')
# subprocess.run(['python', 'ner_eval.py', r'data\ner\dev', r'ner_pred_on_dev_base_model.txt'], check=True)

# #############################POS experiment##############################
pos_train_file = './data/pos/ass1-tagger-train'
model_name = 'roberta-base'
model_params = { 
    'language_model': RobertaForMaskedLM.from_pretrained(model_name),
    'tokenizer': RobertaTokenizerFast.from_pretrained(model_name)
}
pos_train_sentences = utils.get_file_data(pos_train_file)
pos_mdl = models.Bigram_POS_Model()
pos_mdl.fit(pos_train_sentences)
ner_model = models.NER_Model_With_POS_Feats(pos_mdl, model_dict = model_params)
ner_model.fit(normalized_train_data)
y_dev_pred = ner_model.predict(dev_sentences, include_previous_pos=False, inflect_missing=True)
y_dev_pred_with_prev = ner_model.predict(dev_sentences, include_previous_pos=True, inflect_missing=True)
print('\nNER predictions with POS features')
ner_model.save_prediction_to_file(dev_sentences, y_dev_pred, r'ner_pred_on_dev_base_model.txt')
subprocess.run(['python', 'ner_eval.py', r'data\ner\dev', r'ner_pred_on_dev_base_model.txt'], check=True)
print('\nNER predictions with POS features including previous')
ner_model.save_prediction_to_file(dev_sentences, y_dev_pred_with_prev, r'ner_pred_on_dev_base_model.txt')
subprocess.run(['python', 'ner_eval.py', r'data\ner\dev', r'ner_pred_on_dev_base_model.txt'], check=True)

##############################extend dictionary##############################
# model_name = 'roberta-base'
# extension_params = { 
#     'language_model': RobertaForMaskedLM.from_pretrained(model_name),
#     'tokenizer': RobertaTokenizerFast.from_pretrained(model_name),
#     'topk_words': 1000,
#     'topk_sims': 5
# }
# ner_extend_model = models.Base_NER_Model(extension_params)
# ner_extend_model.fit(normalized_train_data)
# y_dev_pred = ner_extend_model.predict(dev_sentences)
# ner_extend_model.save_prediction_to_file(dev_sentences, y_dev_pred, r'ner_pred_on_dev_base_model.txt')
# subprocess.run(['python', 'ner_eval.py', r'data\ner\dev', r'ner_pred_on_dev_base_model.txt'], check=True)

##############################vectors experiment##############################
# model_name = 'roberta-base'
# tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
# roberta_model = RobertaModel.from_pretrained(model_name)
# data_as_vectors = []
# for ner_sentence in normalized_train_data:
#     sentence_as_vectors = []
#     data_as_vectors = []
#     extracted_sentence = ' '.join([word for (word, (bio, ner_tag)) in ner_sentence])
#     tokenized_sentence = tokenizer(extracted_sentence, return_tensors='pt')
#     outputs = roberta_model(**tokenized_sentence)
#     last_hidden_states = outputs.last_hidden_state
#     word_ids = tokenized_sentence.word_ids()
#     for w_loc, word in enumerate(extracted_sentence.split()):
#         word_indices = [i for i, id in enumerate(word_ids) if id==w_loc]
#         word_vector = last_hidden_states[0, word_indices].sum(dim=0, keepdim=True)
#         sentence_as_vectors += [(word, word_vector)]
#     data_as_vectors += sentence_as_vectors   

# pca = decomposition.PCA(n_components=2)
# Z = pca.fit_transform()
