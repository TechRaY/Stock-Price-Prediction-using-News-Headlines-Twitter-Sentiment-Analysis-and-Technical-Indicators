import sys
import re
from nltk.corpus import stopwords
import string
import operator
import random
from sklearn import svm
from nltk.stem import *
import pickle
from sklearn.externals import joblib
from nltk.corpus import wordnet as wn
import unicodedata

f=open(str(sys.argv[1])) #pass filename as parameter
g=open(str('/input/'+sys.argv[1]+'_twt'),'w')

for lines in f:
#  print("proc")
  tweet=lines
  tweet=re.sub('&quot;?',' imp ',tweet)
  tweet=re.sub('&lt;?',' less than ',tweet)
  tweet=re.sub('&gt;?',' greater than ' ,tweet)
  tweet=re.sub('&amp;?',' and ',tweet)
  tweet.strip(' \r\t\n')
  tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
  tweet = re.sub(r'#([^\s])', r'\1', tweet)

#	print tweet

  tweet=tweet.translate(None,'@')
  tweet=re.sub( '\s+', ' ', tweet ).strip()
  tweet=tweet.split()
  st=''
  for words in tweet:
  	if not '@' in words:
  		st=st+words+' '
  st = re.sub(r"([" + re.escape(string.punctuation)+'\w' + r"])\1+", r"\1\1", st)
  text=st.split()

  ret=''
  neg=False

  try:
    for tag in text:
      st=tag.translate(None,string.punctuation).lower()
      syn=[] 
      for synset in wn.synsets(st):
      	for lemma in synset.lemmas():
      		syn.append(lemma.name())
      if len(syn)!=0:
      	st=syn[0]
        st=unicodedata.normalize('NFKD', st).encode('ascii','ignore').translate(None,string.punctuation).lower()
      if neg:
        ret=ret+' not_'+st
      else:
        if tag[len(tag)-3:].lower()=='n\'t':
         ret=ret+' '+tag[:len(tag)-3].translate(None,string.punctuation).lower()
         ret=ret+' not'
        else:
          ret=ret+' '+st
      if any(c in tag for c in string.punctuation):
        neg= False
      if st=='not' or ('n\'t' in tag.lower()) or st=='no':
        neg=not neg
  except UnicodeDecodeError:
    pass


  g.flush()

f.close()
g.close()

sys.exit(0)
