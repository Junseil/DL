import urllib.request
from bs4 import BeautifulSoup

url = 'https://www.imdb.com/title/tt4154796/reviews?ref_=tt_ov_rt%22'
htmlData = urllib.request.urlopen(url)
bs = BeautifulSoup(htmlData, 'lxml')

title_list = bs.findAll('a', 'title')
for title in title_list:
    print(title.getText())

review_list = bs.findAll('div', 'text show-more__control')
for content in review_list:
    print(content.getText() + '\n')
    
score_list = bs.findAll('span', 'rating-other-user-rating')
for score in score_list:
    print(score.span.getText())
    
    
import re

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()