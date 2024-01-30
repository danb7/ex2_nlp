import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
roberta_model = RobertaModel.from_pretrained(model_name)
masked_language_model = RobertaForMaskedLM.from_pretrained(model_name)

text = 'I am so <mask>'

tokenized_text = tokenizer(text, return_tensors='pt')

am_index = tokenized_text["input_ids"][0].tolist().index(tokenizer.encode(" am")[1])
mask_index =  tokenized_text["input_ids"][0].tolist().index(tokenizer.encode(" <mask>")[1])

###############1.1###############
def get_word_vector(word_index, tokenized_sentence, model):
    outputs = model(**tokenized_sentence)
    last_hidden_states = outputs.last_hidden_state
    word_vector = last_hidden_states[0, word_index]
    return word_vector

print('the vector to am')
print(get_word_vector(am_index, tokenized_text, roberta_model))
print('the vector to <mask>')
print(get_word_vector(mask_index, tokenized_text, roberta_model))

###############1.2###############
with torch.no_grad():
    logits = masked_language_model(**tokenized_text).logits

def get_top_k_predictions(logits, word_index, k=5):
    top_k_values, top_k_indices = torch.topk(logits[0, word_index, :], k)
    top_k_probabilities = torch.nn.functional.softmax(top_k_values, dim=-1)
    predicted_tokens = [tokenizer.decode(i) for i in top_k_indices]
    return {'predicted_tokens': predicted_tokens, 'probas': top_k_probabilities}

print("am top-5 predictions and their probabilities:")
am_top_5 = get_top_k_predictions(logits, am_index)
print(list(zip(am_top_5['predicted_tokens'], am_top_5['probas'])))
print("<mask> top-5 predictions and their probabilities:")
mask_top_5 = get_top_k_predictions(logits, mask_index)
print(list(zip(mask_top_5['predicted_tokens'], mask_top_5['probas'])))

###############2###############
similar_sentence_1 = 'I love you'
similar_sentence_2 = 'I love him'
tokenized_similar_sentence_1 = tokenizer(similar_sentence_1, return_tensors='pt')
tokenized_similar_sentence_2 = tokenizer(similar_sentence_2, return_tensors='pt')

love_1_index = tokenized_similar_sentence_1["input_ids"][0].tolist().index(tokenizer.encode(" love")[1])
love_2_index = tokenized_similar_sentence_2["input_ids"][0].tolist().index(tokenizer.encode(" love")[1])
love_1_vector = get_word_vector(love_1_index, tokenized_similar_sentence_1, roberta_model)
love_2_vector = get_word_vector(love_2_index, tokenized_similar_sentence_2, roberta_model)

print('love similarity')
print(cosine_similarity(love_1_vector.detach().numpy().reshape(1, -1), love_2_vector.detach().numpy().reshape(1, -1)))
###############3###############
different_sentence_1 = 'The fission of the cell could be inhibited with certain chemicals.'
different_sentence_2 = 'His cell phone worked, so he spoke with his parents and sister-in-law.'
tokenized_different_sentence_1 = tokenizer(different_sentence_1, return_tensors='pt')
tokenized_different_sentence_2 = tokenizer(different_sentence_2, return_tensors='pt')

cell_1_index = tokenized_different_sentence_1["input_ids"][0].tolist().index(tokenizer.encode(" cell")[1])
cell_2_index = tokenized_different_sentence_2["input_ids"][0].tolist().index(tokenizer.encode(" cell")[1])
cell_1_vector = get_word_vector(cell_1_index, tokenized_different_sentence_1, roberta_model)
cell_2_vector = get_word_vector(cell_2_index, tokenized_different_sentence_2, roberta_model)

print('cell similarity')
print(cosine_similarity(cell_1_vector.detach().numpy().reshape(1, -1), cell_2_vector.detach().numpy().reshape(1, -1)))

###############4###############
sentence_4 = "Didn't I tell you it's gonna be a rock 'n' roll weekend with lots o' fun, and we'll gather 'round the campfire, singin' our favorite songs 'til the break o' dawn?"
tokenized_sentence_4 = tokenizer(sentence_4, return_tensors='pt')
print(f'original sentence: {sentence_4}')
print(f'tokenized sentence: {[tokenizer.decode(t) for t in tokenized_sentence_4["input_ids"][0]]}')
print(f'number of words in sentence: {len(sentence_4.split())}')
print(f'number of tokens in sentence: {len(tokenized_sentence_4["input_ids"][0])}')
