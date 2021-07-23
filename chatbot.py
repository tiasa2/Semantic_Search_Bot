
!pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer

import scipy

model = SentenceTransformer('bert-base-nli-mean-tokens')

import json
t = open('/content/ic.json')
data = json.load(t)
sentences = []
for intents in data['intents']:
  sentences.extend(intents['patterns'])

sentence_embeddings_base = model.encode(sentences)

query = input('What\'s your query? : ')  
queries = [query]
query_embeddings = model.encode(queries)


for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings_base, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    for idx, distance in results[0:1]:
      for intents in data['intents']:
        if intents['patterns'][0] == str(sentences[idx].strip()):
          print(intents['responses'][0])