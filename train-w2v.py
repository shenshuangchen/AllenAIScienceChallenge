from gensim import corpora, models, similarities
import nltk
import os

os.chdir('/Users/mac/OneDrive/courses/COSI-101/term')

##
import pickle

with open('training_set.p', 'r') as f:
	training_set = pickle.load(f)
	
##

def clean(s):
	l = ['.', ':', ',', '!', '?', ';', '"', "'", '(', ')', '[', ']', '{', '}', '-']
	for x in l:
		s = s.replace(x, ' ')
	s = re.sub(r'[^\x00-\x7F]+',' ', s)
	return s
	
##

f = open('Concepts.txt', 'r')
ts = f.readlines()
ts = [x.strip() for x in ts]
ts = [x for x in ts if len(x) > 2]
ts = [x for x in ts if not (x.startswith('Explore') and len(x) < 20)]
ts = [x for x in ts if not (x.startswith('Figure') and len(x) < 20)]
ts = [x for x in ts if not (x.startswith('Review') and len(x) < 20)]
ts = [x for x in ts if not (x.startswith('Summary') and len(x) < 20)]
ts = [x for x in ts if not x.startswith('Use the resource below to answer the questions that follow')]

##
f2 = open('wiki_extract.txt', 'r')
ts2 = f2.readlines()
ts2 = [x.strip() for x in ts2]
ts2 = [x for x in ts2 if len(x) > 2]
ts2 = [x for x in ts if not (x.startswith('</doc'))]
ts2 = [x for x in ts if not (x.startswith('<doc'))]

##

ts = ts + ts2

##

for x in training_set:
	answer = x['answers'][x['answer_id']]
	question = x['question']
	if (question.find('__') == -1):
		ts.append(question + ' ' + answer)
	else:
		ts.append(re.sub('_{2,}', answer, question))
	

##
ts = [clean(x) for x in ts]

##
stemmer = nltk.PorterStemmer()

def stem(l):
	return [stemmer.stem(x.lower()) for x in l]

sentences = [stem(x.split()) for x in ts]


##
pickle.dump(sentences, open('sentences.p', 'w+'))

##


model = models.Word2Vec(sentences, size=300, window=5, min_count=5, workers=4, iter=20)

##
print model.most_similar(positive=['human', 'activ'], negative=['extinct', 'speci'])

##
model.save('w2v_model_context_training_wiki')
