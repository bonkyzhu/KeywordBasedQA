from queue import PriorityQueue
from multiprocessing.pool import ThreadPool
import json
import argparse
import pkuseg

tfidf = json.load(open('rawdata/tfidf.json'))
bm25 = json.load(open('rawdata/bm25.json'))
text = ''
q = PriorityQueue()
method = 'TFIDF'

def seg(article):
  seg = pkuseg.pkuseg()
  return seg.cut(article)

def score(num):
  global text, q, method
  if method == 'BM25':
    tmp = bm25[num]['question']
  if method == 'TFIDF':
    tmp = tfidf[num]['question']
  tmp_score = 0
  for token in text:
    if token in tmp.keys():
      tmp_score += tmp[token]
  q.put((-tmp_score, num))

def QA():
  global text, q
  text = seg(input("Question> "))
  q = PriorityQueue()
  pool = ThreadPool(12)
  pool.map(score, list(tfidf.keys()))
  pool.close()
  num = q.get()[1]
  answer = ' '.join(seg(tfidf[num]['answer']))
  print(f'Answer> {answer}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='manual to this script')
  parser.add_argument('--method', type=str, default = 'BM25')
  args = parser.parse_args()
  method = args.method
  print(f"本次对话使用的是 {method}")
  while True:
    QA()