import lib.models as models
import lib.utils as utils

train_file = './data/pos/ass1-tagger-train'
test_input_file = './data/pos/ass1-tagger-test-input'
dev_input_file = './data/pos/ass1-tagger-dev-input'
dev_file = './data/pos/ass1-tagger-dev'
train_sentences = utils.get_file_data(train_file)
dev_input_sentences = utils.get_file_data(dev_input_file)
dev_sentences = utils.get_file_data(dev_file)
dev_actual = [pos for word, pos in utils.get_all_words_pos(dev_sentences)]
test_sentences = utils.get_file_data(test_input_file)
###1.1###
test_output_file = r'test_predictions\POS_preds_1.txt'
bigram_mdl = models.Bigram_POS_Model()
bigram_mdl.fit(train_sentences)
y_dev_pred = bigram_mdl.bigram_predict(dev_input_sentences)
accuracy = bigram_mdl.calc_Accuracy(y_dev_pred, dev_actual)
print(f"Model: Bigram, Accuracy is: {accuracy}")
y_test_pred = bigram_mdl.bigram_predict(test_sentences)
bigram_mdl.save_prediction_to_file(test_sentences, y_test_pred, test_output_file)
###1.2###
test_output_file = r'test_predictions\POS_preds_2.txt'
static_mdl = models.Static_Vector_POS_Model()
static_mdl.fit(train_sentences)
y_dev_pred = static_mdl.bigram_predict_static(dev_input_sentences)
accuracy = static_mdl.calc_Accuracy(y_dev_pred, dev_actual)
print(f"Model: Bigram with static vectors, Accuracy is: {accuracy}")
y_test_pred = static_mdl.bigram_predict_static(test_sentences)
static_mdl.save_prediction_to_file(test_sentences, y_test_pred, test_output_file)
###1.3###
test_output_file = r'test_predictions\POS_preds_3.txt'
contextualizer_mdl = models.Contextualized_Vector_POS_Model()
contextualizer_mdl.fit(train_sentences)
y_dev_pred = contextualizer_mdl.bigram_predict_with_contextualized(dev_input_sentences)
accuracy = contextualizer_mdl.calc_Accuracy(y_dev_pred, dev_actual)
print(f"Model: Bigram with contextualizer vectors, Accuracy is: {accuracy}")
y_test_pred = contextualizer_mdl.bigram_predict_with_contextualized(test_sentences)
contextualizer_mdl.save_prediction_to_file(test_sentences, y_test_pred, test_output_file)