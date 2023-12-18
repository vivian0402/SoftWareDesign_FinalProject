import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertTokenizerFast, BertModel
from transformers import AlbertModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch.nn.init as nn_init


text1 = 'gender'
text2 = 'sex'
bert_tokenizer = BertTokenizerFast.from_pretrained('./transtab/tokenizer')
mpnet_tokenizer = AutoTokenizer.from_pretrained('./transtab/mpnet_tokenizer')



input1 = bert_tokenizer(text1, truncation=True, padding=True, add_special_tokens=False, return_tensors='pt')
input2 = bert_tokenizer(text2, truncation=True, padding=True, add_special_tokens=False, return_tensors='pt')
word2vec_weight = torch.load('./transtab/bert_emb.pt')
word_embeddings = torch.nn.Embedding.from_pretrained(word2vec_weight, freeze=False, padding_idx=bert_tokenizer.pad_token_id)


output1 = word_embeddings(input1['input_ids'])
output2 = word_embeddings(input2['input_ids'])
output1 = torch.mean(output1, dim=1)
output2 = torch.mean(output2, dim=1)
res = F.cosine_similarity(output1, output2, dim=-1)
print(res)
