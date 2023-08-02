import time
from selenium import webdriver
from bs4 import BeautifulSoup

driver = webdriver.Chrome('./chromedriver')
time.sleep(5)
driver.get('https://www.imdb.com/title/tt4154796/reviews?ref_=tt_ov_rt%27')
