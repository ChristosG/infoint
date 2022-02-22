import gensim
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

filename = './GoogleNews-vectors-negative300.bin.gz'
# Load pretrained model (since inte#rmediate data is not included, the model cannot be refined with additional data)
model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz',binary=True,limit=100000)
model.init_sims(replace=True)
model.save('./bio_word')
model = KeyedVectors.load_word2vec_format(filename, binary=True, limit=100000)
f = open("./embedResults.txt", "w")
f.close()
model = KeyedVectors.load('./bio_word',mmap='r')

# Some predefined functions that show content related information for given words
array =  np.array(model.most_similar("Greece",topn=2))
print(array)

f = open("./embedResults.txt", "w")
# vlepw oti to beasts exei me to creature pou einai kalh ennalaktikh 0.67 sysxetish, Ara to >0.8 den stekei kai poly 
# prepei mhpws na vazoyme ta 2 prwta apotelesmata. isws mono to top?
for x in range(0,2):

        f.write(array[x][0])
        f.write("\n")
f.close()

