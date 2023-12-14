import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertTokenizerFast, BertModel
from transformers import AlbertModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch.nn.init as nn_init

# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# tokenizer.save_pretrained('./test-tokenizer')
# tokenizer = BertTokenizerFast.from_pretrained('./transtab/tokenizer')
# print(tokenizer.vocab_size)

# input = torch.arange(end=tokenizer.vocab_size)
# input = torch.unsqueeze(input, dim=1)
# print(input.shape)

# model = BertModel.from_pretrained('bert-base-uncased')
# model.save_pretrained('./transtab/bert_model')
# model = BertModel.from_pretrained('./transtab/bert_model')

# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
# tokenizer.save_pretrained('./transtab/mpnet_tokenizer')
# model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
# model.save_pretrained('./transtab/mpnet')

# tokenizer = AutoTokenizer.from_pretrained('./transtab/mpnet_tokenizer')
# model = AutoModel.from_pretrained('./transtab/mpnet')

# model = AlbertModel.from_pretrained('albert-base-v2')
# model.save_pretrained('./transtab/albert_model')
# model = AlbertModel.from_pretrained('./transtab/albert_model')

# for name,param in model.named_parameters():
#     # print(name)
#     if name == 'embeddings.word_embeddings.weight':
#         print(param.shape)
#         torch.save(param, "./transtab/mpnet_emb.pt")
#     if name == 'embeddings.LayerNorm.weight':
#         print(param.shape)
#         torch.save(param, "./transtab/mpnet_layernorm_weight.pt")
#     if name == 'embeddings.LayerNorm.bias':
#         print(param.shape)
#         torch.save(param, "./transtab/mpnet_layernorm_bias.pt")

# model.eval()
# with torch.no_grad():
#     output = model(input)
# output = output[0].squeeze(dim=1)
# print(output.shape)
# torch.save(output, './transtab/word2vec.pt')



# text0 = [['i am', 'job is good'], ['tom like', 'i like you forever']]
# text0 = ['']
# text1 = 'sex is male, score is high, country is china'
# text2 = 'gender is male, grade is high'
# input0 = tokenizer(text0, truncation=True, padding=True, add_special_tokens=False, return_tensors='pt')
# input1 = tokenizer(text1, truncation=True, add_special_tokens=False, return_tensors='pt')
# input2 = tokenizer(text2, truncation=True, add_special_tokens=False, return_tensors='pt')

# print(tokenizer.tokenize(text0))
# print(input0)
# print(tokenizer.tokenize(text1))
# print(input1)
# print(tokenizer.tokenize(text2))
# print(input2)

# model.eval()
# with torch.no_grad():
#     output0_word2vec = model(input0['input_ids'])[0]
#     output1_word2vec = model(input1['input_ids'])[0]
#     output2_word2vec = model(input2['input_ids'])[0]
# # print(output1_word2vec.shape)
# # print(output2_word2vec.shape)


# word_embeddings = torch.nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=768, padding_idx=0)
# torch.nn.init.kaiming_normal_(word_embeddings.weight)
# output1_nn = word_embeddings(input1['input_ids'])
# output2_nn = word_embeddings(input2['input_ids'])
# # print(output1_nn.shape)
# # print(output2_nn.shape)


# # output1_word2vec = output1_word2vec.squeeze(dim=0).squeeze(dim=0)
# # output2_word2vec = output2_word2vec.squeeze(dim=0).squeeze(dim=0)
# # output1_nn = output1_nn.squeeze(dim=0).squeeze(dim=0)
# # output2_nn = output2_nn.squeeze(dim=0).squeeze(dim=0)

# # sim_word2vec = torch.dot(F.normalize(output1_word2vec, dim=0), F.normalize(output2_word2vec, dim=0))
# # sim_nn = torch.dot(F.normalize(output1_nn, dim=0), F.normalize(output2_nn, dim=0))
# # print(f'use bert pretrain model : {sim_word2vec}')
# # print(f'use nn.embedding random : {sim_nn}')



# x0 = output1_word2vec[0][0]
# x1 = output2_word2vec[0][0]
# sim_sex = torch.dot(F.normalize(x0, dim=0), F.normalize(x1, dim=0))
# print('..')





























text1 = 'gender'
text2 = 'sex'
bert_tokenizer = BertTokenizerFast.from_pretrained('./transtab/tokenizer')
mpnet_tokenizer = AutoTokenizer.from_pretrained('./transtab/mpnet_tokenizer')



input1 = bert_tokenizer(text1, truncation=True, padding=True, add_special_tokens=False, return_tensors='pt')
input2 = bert_tokenizer(text2, truncation=True, padding=True, add_special_tokens=False, return_tensors='pt')
word2vec_weight = torch.load('./transtab/bert_emb.pt')
word_embeddings = torch.nn.Embedding.from_pretrained(word2vec_weight, freeze=False, padding_idx=bert_tokenizer.pad_token_id)
# word_embeddings = torch.nn.Embedding(bert_tokenizer.vocab_size, 768, padding_idx=bert_tokenizer.pad_token_id)
# nn_init.kaiming_normal_(word_embeddings.weight)

# input1 = mpnet_tokenizer(text1, truncation=True, padding=True, add_special_tokens=False, return_tensors='pt')
# input2 = mpnet_tokenizer(text2, truncation=True, padding=True, add_special_tokens=False, return_tensors='pt')
# word2vec_weight = torch.load('./transtab/mpnet_emb.pt')
# word_embeddings = torch.nn.Embedding.from_pretrained(word2vec_weight, freeze=False, padding_idx=mpnet_tokenizer.pad_token_id)
# word_embeddings = torch.nn.Embedding(mpnet_tokenizer.vocab_size, 768, padding_idx=mpnet_tokenizer.pad_token_id)
# nn_init.kaiming_normal_(word_embeddings.weight)


output1 = word_embeddings(input1['input_ids'])
output2 = word_embeddings(input2['input_ids'])
output1 = torch.mean(output1, dim=1)
output2 = torch.mean(output2, dim=1)
res = F.cosine_similarity(output1, output2, dim=-1)
print(res)

# mpnet
# 0.4379
# random : 0.0445


# bert
# 0.5485
# random : 0.0490




# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# tokenizer = AutoTokenizer.from_pretrained('./transtab/mpnet_tokenizer')
# model = AutoModel.from_pretrained('./transtab/mpnet')

# # Tokenize sentences
# encoded_input1 = tokenizer(text1, padding=True, truncation=True, return_tensors='pt')
# encoded_input2 = tokenizer(text2, padding=True, truncation=True, return_tensors='pt')

# # Compute token embeddings
# with torch.no_grad():
#     model_output1 = model(**encoded_input1)
#     model_output2 = model(**encoded_input2)

# # Perform pooling
# sentence_embeddings1 = mean_pooling(model_output1, encoded_input1['attention_mask'])
# sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask'])
# res = F.cosine_similarity(sentence_embeddings1, sentence_embeddings2, dim=-1)

# print(res)

