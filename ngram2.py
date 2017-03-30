from os import listdir
from os.path import isfile, join
import re
import math
import sys

class NGram:
	def __init__(self,n,total_letter_count, name,smoothing_method):
		self.__grams = {}# a dictionary, for each token, keep a list, first value in it the count, second the probability
		self.__N_freq = {} # keeps how many times each frequency has occured.
		self.__n = n
		self.__name = name
		self.__smoothing_method = smoothing_method
		self.__alpha_normalize = 0.0
		self.__alpha_denom = 1.0
		self.__katz_k = 0
		if n==0:
			self.__total_letter_count = total_letter_count
			#self.__alpha_normalize = 1.0
		else:
			pass

	def set_vocab_size(self,n):
		NGram.vocab_size = n

	def get_vocab_size(self):
		return NGram.vocab_size
		#return len(self.__grams)

	def add_to_the_count(self,token):
		if token in self.__grams:
			self.__grams[token][0]+=1
		else:
			self.__grams[token] = [1]

	def get_the_count(self,token,n):
		if self.__n==0:
			return self.__total_letter_count
		if token in self.__grams:
			return self.__grams[token][0]
		else:
			return 0
			#print('token:',token,'size:',self.__n)
			#raise Exception("Gram was supposed to be available:"+str(token)+"END")

	def compute_probabilities(self,prev_model):
		self.__prev_model = prev_model
		for token in self.__grams:
			partial_token = token[:-1]
			#if partial_token=='':
			#	print('empty partial_token going on', 'token is:',token,'END')
			if self.__smoothing_method=='None':
				if prev_model.get_the_count(partial_token,self.__n-1)==0:
					print('ERROR',self.__grams[token][0],prev_model.get_the_count(partial_token,self.__n-1),token,self.__name)
					sys.stdout.flush()
				self.__grams[token].append( self.__grams[token][0]/prev_model.get_the_count(partial_token,self.__n-1))
			elif self.__smoothing_method=='Laplace':
				if token in self.__grams:
					numerator = self.__grams[token][0] + 1
				else:
					numerator = 1
				denominator = prev_model.get_the_count(partial_token,self.__n-1) + self.get_vocab_size()
				self.__grams[token].append(numerator/denominator)
			else: # Katz
				pass


	def get_probability(self,ngram):
		if self.__smoothing_method=='None':
			if ngram not in self.__grams:
				return 0
			else:
				return self.__grams[ngram][1]

		elif self.__smoothing_method=='Laplace':
			if ngram not in self.__grams:
				partial_token = ngram[:-1]
				#if self.__n==15:
				#	print('Not in the train set, moving to prev_model:',ngram,'prev:',partial_token)
				return 1/(self.__prev_model.get_the_count(partial_token,self.__n-1) + self.get_vocab_size())
			else:
				return self.__grams[ngram][1]

		else:# Katz
			if ngram not in self.__grams:# count == 0, case 1
				partial_token = ngram[1:]
				p2 = self.__prev_model.get_probability(partial_token)
				seen_token = ngram[:-1]
				beta = self.compute_beta(seen_token)
				print('HEY!',self.__prev_model.__alpha_normalize,'prev n is:',self.__n-1)
				#alpha = beta/self.__alpha_normalize
				alpha = beta
				self.__alpha_normalize+= (p2)
				return alpha*p2

			elif self.__grams[ngram][0]<self.__katz_k:# 0<count<k case 2
				count = self.get_the_count(ngram,self.__n)
				Nr_p1 = self.get_N_freq_of_r(count+1)
				Nr = self.get_N_freq_of_r(count)
				return (count * Nr_p1)/(Nr * self.get_total_N_count())
			else:# k<count, case 3
				seen_token = ngram[:-1]
				return self.__grams[ngram][0]/self.__prev_model.get_the_count(seen_token,self.__n-1)

	def update_alpha(self):
		self.__alpha_denom = self.__alpha_normalize

	def get_katz_normalization(self):
		if self.__alpha_normalize == 0.0:
			return 1
		else:
			return self.__alpha_normalize

	def compute_beta(self,seen_token):
		temp= 1- sum([self.get_probability(x) for x in self.__grams if self.__grams[x][0]>self.__katz_k])
		if temp<=0:
			raise Exception('Sum of probability more than 1!',temp)
		return temp

	def set_max_possible_katz_k(self):
		new_k = 1
		while True:
			if new_k not in self.__N_freq:
				break
			new_k +=1
		self.__katz_k= new_k-2 # since each of these need N(r+1), so that should also exist!

	def build_N_freq(self):
		for gram in self.__grams:
			count = self.__grams[gram][0]
			if count in self.__N_freq:
				self.__N_freq[count] +=1
			else:
				self.__N_freq[count]=1
	def print_N_freq(self):
		print('Lang:',self.__name)
		#d = sorted(self.__N_freq,key = lambda x:self.__N_freq[x])
		print(self.__N_freq)
		for key in sorted(self.__N_freq):
			print(key,"seen:",self.__N_freq[key],'bigrams having this frequency')

	def get_N_freq_of_r(self, r):
		if r in self.__N_freq:
			return self.__N_freq[r]
		else:
			return 0

	def get_total_N_count(self):
		total_count = 0
		for key,val in self.__N_freq.items():
			total_count += key*val
		return total_count

	def get_number_of_ngrams(self):
		return len(self.__grams)

	def get_n(self):
		return self.__n

	def get_lang_name(self):
		return self.__name

	def has_ngram(self,ngram):
		if ngram in self.__grams:
			return True
		return False

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
		#if text[ch_ind]==' '
		#if text[ch_ind]==' ' and text[ch_ind-n+1]!= ' ':
		if n!= 1 and text[ch_ind]==' ' and text[ch_ind-n+1:ch_ind+1]!= n * ' ':
			continue
		# get the counts and counts of previous n to have the probability
		token = text[ch_ind-n+1:ch_ind+1]	# app in apple
		# seen_token = text[:] # ap in apple
		ngram.add_to_the_count(token)
	if n==1:
		ngram.set_vocab_size(ngram.get_number_of_ngrams())
	ngram.compute_probabilities(prev_model)
	#print(ngram)
	return ngram

def prepare_text(n , original_content):
	#sep1 = ' '
	sep1 = n* ' '
	#sep2 = sep1
	sep2 = '$' + sep1
	return sep1 + sep2.join(re.sub('[\n+]',' ',original_content).split()) + '$'


def build_lang_models(file_obj,max_n,smoothing_method,lang_name):
	# iterate over n (=n in ngram) to dynamically build them all
	models = []
	original_content = file_obj.read() # keep it since we need it for many times!
	total_letter_count = len(re.sub('[\s+]',' ',original_content)) # how many letters in the text, need it for unigram
	models.append(NGram(0,total_letter_count,lang_name,smoothing_method))
	for n in range(1,max_n+1):
		text = prepare_text(n,original_content)# insert spaces in the text as much as needed
		model = build_ngram(n, text, models[-1],smoothing_method,lang_name)
		models.append(model)
		print('lang:',lang_name,'n in ngram:',n,'number of ngrams seen:',model.get_number_of_ngrams())
		if smoothing_method=='Katz':
			model.build_N_freq()
			model.print_N_freq()
			model.set_max_possible_katz_k()
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
	N = 0
	for ch_ind in range(model.get_n()+1,len(text)):
		token = text[ch_ind-model.get_n()+1:ch_ind+1]
		#if text[ch_ind]==' ' and text[ch_ind-model.get_n()+1]!= ' ':
		#if text[ch_ind]==' ': #and text[ch_ind-model.get_n()+1:ch_ind+1]!= model.get_n() * '':
		#	continue
		if model.get_n()!= 1 and text[ch_ind]==' ' and text[ch_ind-model.get_n()+1:ch_ind+1]!= model.get_n() * ' ':
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
		N +=1
		val = model.get_probability(token)
		if val==0:
			print('ngram not found:',token)
			return 999999
		print('val is',val)
		probability += math.log(val)
		
	#N = len(re.sub('[\s+]',' ',text))
	probability = probability/model.get_katz_normalization()
	perplexity = math.exp((-1*probability)/N)
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

KATZ_K = 5
def __main__():
	# initialization
	train_folder = '650_a3_train'
	dev_folder = '650_a3_dev'
	test_folder = '650_a3_dev'
	#test_folder = '650_a3_test_final'
	max_n = 4
	
	#smoothing_method = 'None'
	#smoothing_method = 'Laplace'
	smoothing_method = 'Katz' 
	
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
			else:
				print('not a match!',item[0].split('/')[1].split('.')[0].split('-')[1])
		print('correct guesses:',correct_counter,'out of:',len(best_found_langs),'for smoothing:',smoothing_method)

	else:
		for item in best_found_langs:
			print(item[0],item[1][0])
	


__main__()