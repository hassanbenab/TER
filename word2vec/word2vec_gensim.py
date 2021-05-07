import json
import pandas as pd
import string
import time

import gensim


import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


sentences=[]
num_lines = sum(1 for line in open('all_sentences.txt',"r", encoding='utf-8'))
f = open("all_sentences.txt", "r", encoding='utf-8')
for i in range(num_lines):
    sentence = f.readline()
    sentences.append(sentence.split())
#-----------------------------------------------------------------------------------------------------
##data = []
##
##for line in open('Cell_Phones_and_Accessories_5.json','r') :
##    data.append(json.loads(line))
##
##
##print(data[0])
##df = pd.DataFrame(data)
##print(len(data))
##
##df.head(10)
##df = df.drop(columns=['reviewerName','style'])
##df1 = df.rename(columns = {'overall': 'rating', 'asin': 'productID'}, inplace = False)
##
##df1.dropna(axis = 0, how ='any',inplace=True) 
##df1.drop_duplicates(subset=['rating','reviewText'],keep='first',inplace=True)
##
##def clean_text(text ): 
##    delete_dict = {sp_character: ' ' for sp_character in string.punctuation} 
##    delete_dict[' '] = ' ' 
##    table = str.maketrans(delete_dict)
##    text1 = text.translate(table)
##    #print('cleaned:'+text1)
##    textArr= text1.split()
##    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>2))]) 
##    
##    return text2.lower().split(' ')
##
##
##df2 = df1.sample(n=2000)
##df2['reviewText']= df2['reviewText'].apply(clean_text)
##
##sentences = df2['reviewText'].tolist()
##
##

#-----------------------------------------------------------------------------------------------------

print(len(sentences))
print(sentences[0])
#print(sentences[200])
    
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec
# init callback class
class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        elif self.epoch % 100 == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        
        
        self.epoch += 1
        self.loss_previous_step = loss





# init word2vec class
w2v_model = Word2Vec(vector_size = 300,
                     window = 3,
                     min_count = 2,
                     workers = 20,
                     sg = 1,
                     negative = 5,
                     sample = 1e-5)
# build vocab


w2v_model.build_vocab(sentences)

  
# train the w2v model
start = time.time()
w2v_model.train(sentences, 
                total_examples=w2v_model.corpus_count, 
                epochs=1001,
                report_delay=1,
                compute_loss = True, # set compute_loss = True
                callbacks=[callback()]) # add the callback class
end = time.time()


print("elapsedtime in seconds :"+ str(end - start))
# save the word2vec model
w2v_model.save('word2vec.model')


reloaded_w2v_model = Word2Vec.load('word2vec.model')
words = list(reloaded_w2v_model.wv.index_to_key)
print('Vocab size: '+str(len(words)))
w1 = 'trimestre'
print("Top 3 words similar to trimestre:",\
      reloaded_w2v_model.wv.most_similar(positive = w1,topn =3))
w1 = 'travail'
print("Top 3 words similar to travail:",\
      reloaded_w2v_model.wv.most_similar(positive = w1,topn =3))
print("Similarity between travail and sérieux:"+\
      str(reloaded_w2v_model.wv.similarity(w1="travail",w2="sérieux")))
print("Similarity between excellent and correct:"+\
      str(reloaded_w2v_model.wv.similarity(w1="excellent",w2="correct")))






def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = [] # positions in vector space
    labels = [] # keep track of words to label our data again later
    for word in model.wv.index_to_key:
        vectors.append(model.wv[word])
        labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    #labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


x_vals, y_vals, labels = reduce_dimensions(reloaded_w2v_model)



def plot_with_matplotlib(x_vals, y_vals, labels):


    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    
    
    indices = list(range(len(labels)))
    #selected_indices = random.sample(indices, 25)
    selected_indices=[]
    index = labels.index("trimestre")
    selected_indices.append(index)
    index = labels.index("sérieux")
    selected_indices.append(index)
    index = labels.index("correct")
    selected_indices.append(index)
    index = labels.index("excellent")
    selected_indices.append(index)
    index = labels.index("progrès")
    selected_indices.append(index)
    index = labels.index("travail")
    selected_indices.append(index)
    index = labels.index("difficulté")
    selected_indices.append(index)
    index = labels.index("catastrophique")
    selected_indices.append(index)
    index = labels.index("satisfaisants")
    selected_indices.append(index)
    index = labels.index("niveau")
    selected_indices.append(index)
    index = labels.index("classe")
    selected_indices.append(index)

    index = labels.index("niveau")
    selected_indices.append(index)
    index = labels.index("orientation")
    selected_indices.append(index)
    index = labels.index("attitude")
    selected_indices.append(index)
    index = labels.index("régulier")
    selected_indices.append(index)
    index = labels.index("problèmes")
    selected_indices.append(index)
    index = labels.index("lacunes")
    selected_indices.append(index)

    
    
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))



plot_function = plot_with_matplotlib

plot_function(x_vals, y_vals, labels)
plt.show()

