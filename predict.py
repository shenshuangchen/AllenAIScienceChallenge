model = gensim.models.Word2Vec.load('w2v_model_context_training')

## test 
import operator
import sys

def get_correctness(data, sigma ):
	corrent_count = 0;
	i = 0
	for x in data:
		#print i
		i += 1
		#question_v = getVec(x['question'], use_keyword = False)
		#answers_v = [getVec(y, use_keyword = False) for y in x['answers']]
		#answers_s = [np.dot(y, question_v) for y in answers_v]
		#max_index, max_value = max(enumerate(answers_s), key=operator.itemgetter(1))
		#max_index = x_result(x['question'], x['answers'])
		max_index = x_gaussian_predict(x['question'], x['answers'], sigma)
		if (max_index == x['answer_id']):
			corrent_count += 1
	return float(corrent_count)/len(data)

##
print(get_correctness([x for x in training_set if x['question'].find("not") != -1], r = 1))

##
r = 0.0
step = 0.05
while r < 1:
	print('r: ' + str(r) + ', correctness: ' + str(get_correctness(training_set, r)))
	r += step
	
## output
import operator
import sys

m = ['A', 'B', 'C', 'D']

output = []

i = 0
for x in validation_set:
	print(i)
    i += 1
	max_index = x_gaussian_predict(x['question'], x['answers'], 0.65)
    output.append({'id': x['id'], 'correctAnswer': m[max_index]})


##
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

import string


def parse_sentence(s):
    s.replace('-', ' ')
    s = re.sub(r'[^\x00-\x7F]+',' ', s)
	ss = sent_detector.tokenize(s.strip())
	ss = [s.translate(None, string.punctuation) for s in ss]
	ss = [s.split() for s in ss]
	#ss = [[w for w in s if w not in stopwords] for s in ss]
	ss = [[stemmer.stem(w.lower()) for w in s] for s in ss]
	ss = [[w for w in s if w in model] for s in ss]
	positive = []
	negative = []
	for s in ss:
		neg = False
		for w in s:
			if w == 'cannot' or w == 'not':
				neg = True
				continue
			if neg:
				negative.append(w)
			else:
				positive.append(w)
	positive = [w for w in positive if w not in stopwords]
	negative = [w for w in negative if w not in stopwords]
	return positive, negative
	
def parse_answer(s):
    s.replace('-', ' ')
    s = re.sub(r'[^\x00-\x7F]+',' ', s)
	s = s.translate(None, string.punctuation)
	s = s.split()
	s = [stemmer.stem(w.lower()) for w in s]
	s = [w for w in s if w not in stopwords and w in model]
	return s
		
##

	
question = [x['question'] for x in training_set if x['question'].find("not") != -1][3]
answers = [x['answers'] for x in training_set if x['question'].find("not") != -1][3]
#positive, negative = parse_sentence(question)
#answers = [parse_answer(x) for x in answers]

##
def x_result(question, answers, r=0.8, default_s=0):
	answers = [parse_answer(x) for x in answers]
	positive, negative = parse_sentence(question)
	positve_score = None
	negatve_score = None
	if len(positive) != 0:
		positve_score = [model.n_similarity(positive, l) if len(l) != 0 else default_s for l in answers]
	if len(negative) != 0:
		negatve_score = [1-model.n_similarity(negative, l) if len(l) != 0 else default_s for l in answers]
	target = None
	if positve_score != None and negatve_score != None:
		target = [(1-r)*x+r*y for x,y in zip(positve_score, negatve_score)]
	if positve_score != None:
		target = positve_score
	if negatve_score != None:
		target = negatve_score
	#print answers
	max_index, max_value = max(enumerate(target), key=operator.itemgetter(1))
	return max_index
	
##
def mean(s):
    l = s.split()
    l = stem(l)
    vecs = []
    for x in l:
        if len(x) == 0 or x not in model or x in stopwords:
            continue
        vecs.append(model[x])
    if len(vecs) == 0:
        return None
    else:
        return np.array(vecs, dtype=np.float32).mean(axis=0)
        
def getVec(s, use_keyword = True, default_vec = np.array([0.5 for i in range(300)])):
    s.replace('-', ' ')
    s = re.sub(r'[^\x00-\x7F]+',' ', s)
    if not use_keyword:
        vec = mean(s)
        return vec if vec != None else default_vec
    keywords = rake.extract(s, incl_scores=True)
    keywords = [(mean(w[0]), w[1]) for w in keywords]
    keywords = [w for w in keywords if w[0] != None]
    total_score = sum([w[1] for w in keywords])
    #keywords = [w[0] * (w[1] / total_score) for w in keywords]
    #return sum(keywords) if len(keywords) != 0 else default_vec
    keywords = [w[0] for w in keywords]
    return np.array(keywords).mean(axis=0) if len(keywords) != 0 else default_vec


##
def gaussian(x, mu, sig):
	return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def x_gaussian_predict(question, answers, sig):
	#print(question)
	s = question
	s = s.replace('-', ' ')
	s = re.sub(r'[^\x00-\x7F]+',' ', s)
	s = s.strip()
	s = s.translate(None, string.punctuation.replace('_',''))
	s = re.sub('_{2,}', '___', s)
	s = s.split()
	s = [stemmer.stem(w.lower()) for w in s]
	keys = ['what','of','when','is','which','to','how','why','are','where','would','because','for', '___']
	distances = []
	length = len(s)
	distances.append([1.0/length for j in range(length)])
	for i in range(len(s)):
		if s[i] not in keys:
			continue
		distances.append([abs(j-i) for j in range(len(s))])
	distances = [[gaussian(d, 0, sig) for d in distance] for distance in distances]
	weights = []
	words = []
	for i in range(len(s)):
		if s[i] not in model or s[i] in stopwords:
			continue
		words.append(s[i])
		weights.append(sum([w[i] for w in distances]))
	weights_sum = sum(weights)
	weights = [w/weights_sum for w in weights]
	question_v = sum([model[word]*weight for word, weight in zip(words, weights)])
    answers_v = [getVec(y, use_keyword = False) for y in answers]
    answers_s = [np.dot(y, question_v) for y in answers_v]
    max_index, max_value = max(enumerate(answers_s), key=operator.itemgetter(1))
	return max_index

## 
# context + training + wiki        0.3388
# context + training               0.3588
# context                          0.3144
# context + wiki                   0.31
# context + wiki + not             0.3344
# context + training + not         0.3704
# context + training + wiki + not  0.3732