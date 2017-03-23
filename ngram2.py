from os import listdir
from os.path import isfile, join
import re
import math

class NGram:
	def __init__(self,n,total_letter_count, name,smoothing_method):
		self.__grams = {}# a dictionary, for each token, keep a list, first value in it the count, second the probability
		self.__n = n
		self.__name = name
		self.__smoothing_method = smoothing_method
		if n==0:
			self.__total_letter_count = total_letter_count
		else:
			pass

	def set_vocab_size(self,n):
		NGram.vocab_size = n

	def get_vocab_size(self):
		return NGram.vocab_size

	def add_to_the_count(self,token):
		if token in self.__grams:
			self.__grams[token][0]+=1
		else:
			self.__grams[token] = [1]

	def get_the_count(self,token,n):
		if n==0:
			return self.__total_letter_count
		if token in self.__grams:
			return self.__grams[token][0]
		else:
			return 0
			#print('token:',token,'size:',self.__n)
			#raise Exception("Gram was supposed to be available:"+str(token)+"END")

	def compute_probabilities(self,prev_model,smoothing_method):
		self.__prev_model = prev_model
		for token in self.__grams:
			partial_token = token[:-1]
			#if partial_token=='':
			#	print('empty partial_token going on', 'token is:',token,'END')
			if smoothing_method=='None':
				self.__grams[token].append( self.__grams[token][0]/prev_model.get_the_count(partial_token,self.__n-1))
			elif smoothing_method=='Laplace':
				if token in self.__grams:
					nominator = self.__grams[token][0] + 1
				else:
					nominator = 1
				denominator = prev_model.get_the_count(partial_token,self.__n-1) + self.get_vocab_size()
				self.__grams[token].append(nominator/denominator)
			else: # Katz
				pass

	def get_n(self):
		return self.__n

	def get_lang_name(self):
		return self.__name

	def has_ngram(self,ngram):
		if ngram in self.__grams:
			return True
		return False

	def get_probability(self,ngram):
		if ngram not in self.__grams:
			#raise Exception('why are you asking for an unavailable ngram??')
			if self.__smoothing_method=='None':
				return 0
			elif self.__smoothing_method=='Laplace':
				partial_token = ngram[:-1]
				return 1/(self.__prev_model.get_the_count(partial_token,self.__n-1) + self.get_vocab_size())
			else:# Katz
				pass

		return self.__grams[ngram][1]

	def get_number_of_letters(self):
		return len(self.__grams)

	def __str__(self):
		ret_str = 'n = ' + str(self.__n) + '\n'
		for item in self.__grams:
			if len(self.__grams[item])>1:
				ret_str+=str(item) + ' ' + str(self.__grams[item][0]) + ' ' + str(self.__grams[item][1]) + '\n'
			else:
				ret_str+=str(item) + ' ' + str(self.__grams[item][0]) + '\n'
		return ret_str

############################# TRAIN PART ###############################


def build_ngram(n,text,prev_model,smoothing_method,lang_name):
	# count them,
	ngram = NGram(n,0,lang_name,smoothing_method)# the second argument is only for n=0
	for ch_ind in range(n,len(text)):
		if text[ch_ind]==' ' and text[ch_ind-n+1]!= ' ':
			continue
		# get the counts and counts of previous n to have the probability
		token = text[ch_ind-n+1:ch_ind+1]	# app in apple
		# seen_token = text[:] # ap in apple
		ngram.add_to_the_count(token)
	if n==1:
		ngram.set_vocab_size(ngram.get_number_of_letters())
	ngram.compute_probabilities(prev_model,smoothing_method)
	#print(ngram)
	return ngram

def prepare_text(n , original_content):
	sep1 = n* ' '
	sep2 = '$' + sep1
	return sep1 + sep2.join(original_content.split()) + '$'


def build_lang_models(file_obj,max_n,smoothing_method,lang_name):
	# iterate over n (=n in ngram) to dynamically build them all
	models = []
	original_content = file_obj.read() # keep it since we need it for many times!
	total_letter_count = len(re.sub('[\s+]','',original_content)) # how many letters in the text, need it for unigram
	models.append(NGram(0,total_letter_count,lang_name,smoothing_method))
	for n in range(1,max_n+1):
		text = prepare_text(n,original_content)# insert spaces in the text as much as needed
		model = build_ngram(n, text, models[-1],smoothing_method,lang_name)
		models.append(model)
	return models


def train(train_folder,max_n,smoothing_method):
	# for each file, separately create ngrams and store them in a dictionary
    files = [join(train_folder, f) for f in listdir(train_folder) if isfile(join(train_folder, f))]
    lang_models = {}
    for file in files:
    	lang = file.split('.')[0].split('-')[1]
    	file_obj = open(file,'r')
    	lang_models[lang] = build_lang_models(file_obj,max_n,smoothing_method,lang)
    	file_obj.close()
    return lang_models

############################# DEV PART ###############################
def get_perplexity(model, text):
	probability = 0.0
	for ch_ind in range(model.get_n(),len(text)):
		ngram = text[ch_ind-model.get_n()+1:ch_ind+1]
		if text[ch_ind]==' ' and text[ch_ind-model.get_n()+1]!= ' ':
			continue
		'''
		if model.has_ngram(ngram):
			val = model.get_probability(ngram)
			if val==0:
				#print('ngram not found:',ngram)
				return 999999
			probability += math.log(val)
			#print(probability)
		else:
			print('No engram detected!')
		'''
		val = model.get_probability(ngram)
		if val==0:
			#print('ngram not found:',ngram)
			return 999999
		probability += math.log(val)
		

	perplexity = math.pow(2,(-1*probability)/(len(text) - model.get_n()))
	return perplexity

def evaluate_dev_file(models,file_obj):
	original_content = file_obj.read()
	results = []
	for n in range(1,len(models)):
		# for each model(unigram, bigran, ...) create the appropriate text and then 
		text = prepare_text(n,original_content)# insert spaces in the text as much as needed
		result = get_perplexity(models[n],text)
		results.append(result)
	return results
		
def dev(dev_folder,lang_models):
	files = [join(dev_folder, f) for f in listdir(dev_folder) if isfile(join(dev_folder, f))]
	best_models_indices = {}
	for file in files:
		lang = file.split('.')[0].split('-')[1]
		if lang not in lang_models:
			raise Exception('language does not exist!!')
		models = lang_models[lang]
		file_obj = open(file,'r')
		result = evaluate_dev_file(models, file_obj)
		file_obj.close()
		print('language:',lang,result)# may need to print it in a better way!
		min_perplexity = 999
		min_perplexity_ind = -1
		for res_ind in range(len(result)):
			if result[res_ind]<min_perplexity:
				min_perplexity = result[res_ind]
				min_perplexity_ind = res_ind 
		best_models_indices[lang] = min_perplexity_ind + 1 # since they start from 1, need to be shifted by one
	return best_models_indices

############################# TEST PART ###############################

def evaluate_test_file(best_models, file_obj): # content exactly same as dev.
	original_content = file_obj.read()
	results = []
	for model in best_models:
		# for each model(unigram, bigran, ...) create the appropriate text and then 
		text = prepare_text(best_models[model].get_n(),original_content)# insert spaces in the text as much as needed
		result = get_perplexity(best_models[model],text)
		results.append((best_models[model].get_lang_name(),result))
	return results


def test(test_folder,lang_models, best_models_indices):
	files = [join(test_folder, f) for f in listdir(test_folder) if isfile(join(test_folder, f))]
	results = []
	best_models = {}
	for lang_name in lang_models:
		best_models[lang_name] = lang_models[lang_name][best_models_indices[lang_name]]
	# now for each language we have its best model!
	predicted_langs = {}
	for file in files:
	# for each file, compute perplexity of each given model
		file_obj = open(file,'r')
		predicted_lang = evaluate_test_file(best_models,file_obj) # should return name of the file, and its perplexility in a tuple!
		file_obj.close()
		predicted_langs[file] = predicted_lang # for each file, I have its predicted lang name, and perplexity, may need to change based on the question description
		temp = sorted(predicted_lang,key=lambda x:x[1])
		print('lang:', file)
		print(temp)
		print()
		results.append((file,temp[0],temp[1],temp[2]))
	return results

############################### main ###############################

def __main__():
	# initialization
	train_folder = '650_a3_train'
	dev_folder = '650_a3_dev'
	test_folder = '650_a3_dev'
	max_n = 15
	
	#smoothing_method = 'None'
	smoothing_method = 'Laplace'
	# TODO:
	#smoothing_method = 'Katz' 
	
	# for each language, we need ngrams for n = [1,2,...], then test!
	lang_models = {} # key=language, value=list of ngrams, where item 1 is unigram and so on, each of them a obj is kept.
	
	# training
	lang_models = train(train_folder,max_n,smoothing_method)

	# use development set to find best Ns for each language. Also may need to do sth for smoothing method
	best_models_indices = dev(dev_folder,lang_models)
	print('best models (# in n in ngram):', best_models_indices)

	# test
	# read all files in the test folder, and find the model that has the least perplexity in them.
	best_found_langs = test(test_folder,lang_models,best_models_indices)
	# also for case of dev, can check how many of them are giving correct results.
	correct_counter = 0
	if dev_folder==test_folder:
		for item in best_found_langs:
			if item[1][0]==item[0].split('/')[1].split('.')[0].split('-')[1]:
				correct_counter+=1
		print('correct guesses:',correct_counter,'out of:',len(best_found_langs),'for smoothing:',smoothing_method)



__main__()