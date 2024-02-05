import subprocess
import utils
import models


train_file = r'data\ner\train'
dev_file = r'data\ner\dev'

normalized_train_data = utils.get_ner_data(train_file)
dev_sentences = utils.get_ner_data_with_no_tags(dev_file)

ner_base_model = models.Base_NER_Model()
ner_base_model.fit(normalized_train_data)
y_dev_pred = ner_base_model.predict(dev_sentences)

ner_base_model.save_prediction_to_file(dev_sentences, y_dev_pred, r'ner_pred_on_dev_base_model.txt')
subprocess.run(['python', 'ner_eval.py', r'data\ner\dev', r'ner_pred_on_dev_base_model.txt'], check=True)