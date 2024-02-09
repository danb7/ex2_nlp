import subprocess

from transformers import RobertaTokenizerFast, RobertaForMaskedLM
import lib.utils as utils
import lib.models as models

# files definitions
train_file = r'data\ner\train'
dev_file = r'data\ner\dev'
test_file = r'data\ner\test.blind'

# reading data
normalized_train_data = utils.get_ner_data(train_file)
dev_sentences = utils.get_ner_data_with_no_tags(dev_file)
test_sentences = utils.get_file_data(test_file, as_lists=True)

# pretrain a POS Model
pos_train_file = './data/pos/ass1-tagger-train'
pos_train_sentences = utils.get_file_data(pos_train_file)
pos_mdl = models.Bigram_POS_Model()
pos_mdl.fit(pos_train_sentences)

# define a contextualized word vectors model
model_name = 'roberta-base'
model_params = { 
    'language_model': RobertaForMaskedLM.from_pretrained(model_name),
    'tokenizer': RobertaTokenizerFast.from_pretrained(model_name)
}

# fitting our best model
ner_model = models.NER_Model_With_POS_Feats(pos_mdl, model_dict = model_params)
ner_model.fit(normalized_train_data)

# predicting on dev for performance evaluation
print('Predicting dev set NER tags...')
y_dev_pred = ner_model.predict(dev_sentences, include_previous_pos=True, inflect_missing=True)
print('\nNER predictions evaluation on dev set:')
ner_model.save_prediction_to_file(dev_sentences, y_dev_pred, r'temp\ner_pred_on_dev_best_model.txt')
subprocess.run(['python', 'ner_eval.py', r'data\ner\dev', r'temp\ner_pred_on_dev_best_model.txt'], check=True)

# predicting on test and saving file
print('Predicting test set NER tags...')
y_test_pred = ner_model.predict(test_sentences, include_previous_pos=True, inflect_missing=True)
ner_model.save_prediction_to_file(test_sentences, y_test_pred, r'test_predictions\NER_preds.txt')
