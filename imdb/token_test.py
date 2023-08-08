import nltk

nltk.download('punkt')

sentence = '''
I have a dream
that my four little children will one day live in a nation
where they will not be judged by the color of their skin
but by the content of their character.
'''
sentence = sentence.lower()
tokens = nltk.word_tokenize(sentence)

text = nltk.Text(tokens)

for token in text.vocab():
    print(token, text.vocab()[token])

print(text.vocab())
# text.plot()

