import urllib.request
from bs4 import BeautifulSoup

url = 'https://www.imdb.com/title/tt4154796/reviews?ref_=tt_ov_rt%22'
htmlData = urllib.request.urlopen(url)
bs = BeautifulSoup(htmlData, 'lxml')

title_list = bs.findAll('a', 'title')
review_list = bs.findAll('div', 'text show-more__control')
score_list = bs.findAll('span', 'rating-other-user-rating')
    
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

with open('./review.txt', 'w', encoding='UTF-8') as f:
    for i in range(len(title_list)):
        line = clean_str(title_list[i].getText()) + ' ' + clean_str(review_list[i].getText()) + '\n'
        f.write(line)

with open('./score.txt', 'w', encoding='UTF-8') as f:
    for i in range(len(score_list)):
        line = score_list[i].span.getText() + '\n'
        f.write(line)
        