
# this is a simple example of using the gensim library
# to do a word2vec embedding

# this example is adopted from Dr. Joseph Barr at SDSU


from gensim.models import Word2Vec

sentences = [
    ['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
    ['this', 'is', 'the', 'second', 'sentence'],
    ['yet', 'another', 'sentence'],
    ['one', 'more', 'sentence'],
    ['and', 'the', 'final', 'sentence']
]

model = Word2Vec(sentences, min_count = 1)

print(model)

model.wv.vocab # gives a dictionary
list(model.wv.vocab)  # gives the actual words

# most similar words to 'sentence'
model.wv.most_similar('sentence')

# Get the probability distribution of the center word given context words
model.predict_output_word(['this', 'is'], topn = 2)


# get this error when using 'impor word2vec' with lowercase 'w' and 'v'
# because this is how the module is written in the documentation
# https://radimrehurek.com/gensim/models/word2vec.html


# =============================================================================
# model = word2vec(sentences)Traceback (most recent call last):
# 
#   File "<ipython-input-2-59b43a919d0a>", line 1, in <module>
#     model = word2vec(sentences)
# 
# TypeError: 'module' object is not callable
# =============================================================================

# there wasn't an error on importing with lowercase 'w' and 'v'
# so everything appeared to be working fine
# however, when calling Word2Vec() with lowercase 'w' and 'v',
#  there was a complaint about it not finding the function
# when importing with capital 'W' and 'V', and using same in function call,
# it worked fine
# really, this is a documentation issue



