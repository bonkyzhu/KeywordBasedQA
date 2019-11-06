import json
from gensim import corpora, models, similarities
from gensim.summarization.bm25 import *
import pkuseg
from math import sqrt

train_file = 'iqa.train.json'
# train_file = 'train.json'
test_file = 'iqa.test.json'
valid_file = 'iqa.valid.json'
vocab_file = 'iqa.vocab.json'

'''
vocab_data包含word2id(dict, 从word到id), id2word(dict, 从id到word),tf(dict, 词频统计)和total(单词总数)。 
其中，未登录词的标识为UNKNOWN，未登录词的id为0。
'''
vocab = json.load(open(vocab_file))

'''
train_data, test_data 和 valid_data 的数据格式一样。
qid 是问题Id，question 是问题，utterance 是回复，
label 如果是 [1,0] 代表回复是正确答案，[0,1] 代表回复不是正确答案，所以 utterance 包含了正例和负例的数据。
每个问题含有10个负例和1个正例。
'''
train = json.load(open(train_file))
test = json.load(open(test_file))
valid = json.load(open(valid_file))

i = 0

def seg(article):
  seg = pkuseg.pkuseg()
  return seg.cut(article)

def to_text(data, type):
  text = {}
  global i
  for line in data:
    if line['label'] == [1,0]:
      t = {}
      t['question'] = [vocab['id2word'][str(id)] for id in line['question']]
      t['answer'] = [vocab['id2word'][str(id)] for id in line['utterance']]
      text[str(i)] = (t)
      i += 1
  json.dump(text, open(type+'.json', 'w', encoding='utf-8'), ensure_ascii=False)
  print(f"{type} Finished")

def get_texts(data):
  texts = []
  answer = []
  for line in data:
    if line['label'] == [1,0]:
      t = [vocab['id2word'][str(id)] for id in line['question']]
      ans = [vocab['id2word'][str(id)] for id in line['utterance']]
      texts.append(t)
      answer.append(ans)
  return texts, answer

def to_dict(tfidf, corpus, answer, dictionary):
  the_dict = {}
  corpus_tfidf = tfidf[corpus]
  for i, doc in enumerate(corpus_tfidf):
    t = {}
    t['question'] = {dictionary[word[0]]:word[1] for word in doc}
    t['answer'] = ''.join(answer[i])
    the_dict[str(i)]= t
  return the_dict

def to_corpus(data, type):
  texts,answer = get_texts(data)
  print(len(texts))
  dictionary = corpora.Dictionary(texts)
  corpus = [dictionary.doc2bow(text) for text in texts]
  tfidf = models.TfidfModel(corpus)
  the_dict = to_dict(tfidf, corpus, answer, dictionary)
  json.dump(the_dict, open(type+'.json', 'w', encoding='utf-8'), ensure_ascii=False)
  print(f"{type} Finished")
  return the_dict, texts, answer

def to_TF(article):
  '''
  计算文章/句子所有词的TF值，返回一个字典
  :param article: str
  :return article_TF: dict
  '''
  length = len(article)
  article_TF = {}
  for token in article:
    if token not in article_TF.keys():
      article_TF[token] = 1
    else:
      article_TF[token] += 1
  for key in article_TF.keys():
    article_TF[key] /= length
    article_TF[key] = round(sqrt(article_TF[key]), 4)
  return article_TF

def BM25_Algorithm(article, avgDl, corpus_IDF, k=2, b=0.75):
  '''
  :param k, b
         d: 当前文章的长度
         avgdl当前文章的长度
  '''
  text = set(article)
  d = len(text)
  similarity = {}
  article_TF = to_TF(article)
  for token in text:
    idf = corpus_IDF[token]
    tf = article_TF[token]
    similarity[token] = idf * ((k + 1) * tf) / (k * (1.0 - b + b * (d/avgDl)) + tf)
    similarity[token] = round(similarity[token], 4)
  return similarity

def bm25(texts, answer):
  bm25_object = BM25(texts)
  bm25 = {}
  avgDl = bm25_object.avgdl
  corpus_IDF = bm25_object.idf
  for index, question in enumerate(texts):
    t = {}
    t["question"] = BM25_Algorithm(question, avgDl, corpus_IDF)
    t["answer"] = ' '.join(answer[index])
    bm25[str(index)] = t
  json.dump(bm25, open('bm25.json', 'w', encoding='utf-8'), ensure_ascii=False)
  print("bm25 Finished")
  return bm25


if __name__ == '__main__':
  texts,answer = get_texts(train)
  bm25(texts, answer)
  to_corpus(train, 'tfidf')
  # to_corpus(test, 'test')
  # to_corpus(valid, 'valid')

