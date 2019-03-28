
# this is a simple look into the gensim package
# here we look at the first section (chapter) of Flatland


import re
from gensim.models import Word2Vec

file_path = '/Users/andrewbates/Desktop/neural-nets/data/flatland-section-1.txt'

flatland_file = open(file_path, 'r')
raw_text = flatland_file.read()
flatland_file.close()

# the file is formatted like a book with paragraphs
#  it has new lines mixed in with sentences, probably to fit in a single page
# there are also other things like commas to remove
# there may be a way to do this with one regex but i don't know how
# also, gensim expects a list of lists where the outer list is the corpus
#  and the inner lists are the documents, tokenized into words
flatland = re.sub('\n', ' ', raw_text)
flatland = re.sub(',', '', flatland)
flatland = re.sub('"', '', flatland)
flatland = re.sub(':', '', flatland)
flatland = [word.lower().split(' ') for word in flatland.split('.')]

# there are only ~4,000 words total so 
#  keep words that appear twice instead of the default 5
#  and use 10-d vectors instead of the default 100-d
cbow_model = Word2Vec(flatland, size = 10, min_count = 2)

skipgram_model = Word2Vec(flatland, size = 10, min_count = 2, sg = 1)

list(cbow_model.wv.vocab)

# get the vector corresponding to 'triangles'
cbow_model.wv.word_vec('triangles')
skipgram_model.wv.word_vec('triangles')

# what words are similar to 'triangles'?
# why is 'squares' not similar?! what if we used the full text?
cbow_model.wv.most_similar('triangles')
skipgram_model.wv.most_similar('triangles')

# at least 'triangles' and 'squares' are both similar to other words
cbow_model.wv.most_similar('squares')
skipgram_model.wv.most_similar('squares')

# there are both singular and plural forms 
cbow_model.wv.most_similar('triangle')
skipgram_model.wv.most_similar('triangle')

cbow_model.wv.most_similar('square')
skipgram_model.wv.most_similar('square')


# how similar are 'triangles' and 'squares'
cbow_model.wv.similarity('triangle', 'square')
skipgram_model.wv.similarity('triangle', 'square')



