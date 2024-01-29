import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)
masked_language_model = RobertaForMaskedLM.from_pretrained(model_name)


text = 'I am so <mask>'

tokenized_text = tokenizer(text, return_tensors='pt')

outputs = model(**tokenized_text)
last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.shape)
# print(last_hidden_states)
# print(tokenizer.decode(424))
# print(tokenizer.decode(524))
am_index = tokenized_text["input_ids"][0].tolist().index(tokenizer.encode(" am")[1])
mask_index =  tokenized_text["input_ids"][0].tolist().index(tokenizer.encode(" <mask>")[1])
am_vector = last_hidden_states[0, am_index]
mask_vector = last_hidden_states[0, mask_index]
print('the vector to am')
print(am_vector)
print('the vector to <mask>')
print(mask_vector)


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

################2###############