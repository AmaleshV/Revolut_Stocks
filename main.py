# Core Pkgs

import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from tqdm import tqdm
from datetime import datetime
from datetime import date, timedelta
# Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import base64
import seaborn as sns 
import pandas as pd
import numpy as np
import random
import sklearn
import re
import time
#from dfply import *
from datetime import date, timedelta
import numpy as np
# To extract fundamental data
from bs4 import BeautifulSoup as bs
import requests
import numpy as np
from tqdm import tqdm
import pandas as pd
#https://towardsdatascience.com/sentiment-analysis-of-stocks-from-financial-news-using-python-82ebdcefb638
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import os
import pandas as pd
import matplotlib.pyplot as plt
# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
from bs4 import BeautifulSoup as bs
import requests
from io import BytesIO
class tqdm:
    def __init__(self, iterable, title=None):
        if title:
            st.write(title)
        self.prog_bar = st.progress(0)
        self.iterable = iterable
        self.length = len(iterable)
        self.i = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.i += 1
            current_prog = self.i / self.length
            self.prog_bar.progress(current_prog)




def main():
	"""Semi Automated ML App with Streamlit """

	st.write("*Revolut Daily Stock changes*")		
	no_days= st.slider('How many last days News to do sentiment analysis?', 1,7, step=1 )	
	threshold=st.slider('Threshold sentiment stocks to be selected',0.0,1.0, step=0.1)

	data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
	if data is not None:
		stock_list = pd.read_csv(data,encoding='utf-8')

		st.dataframe(stock_list['Ticker'].head())
		

		st.write("Number of Stocks")

		st.write(len(stock_list['Ticker']))

		st.write("Stocks Pre-Processing")
		stock_list=stock_list[stock_list.Ticker != 'BF.B']
		stock_list=stock_list[stock_list.Ticker != 'BMY.RT']
		stock_list=stock_list[stock_list.Ticker != 'BRK.B']
						  				 
		st.dataframe(stock_list)
		st.write(len(stock_list["Ticker"]))

		# if st.checkbox("Daily revolut details"):
		# 				# =============================================================================
		# 	stock= pd.DataFrame() 
		# 	stock['ticker']=''
		# 	stock['perc']= ''
		# 	stock['close']= ''
		# 	stock['last_trade']=''
		# 	stock['change'] =''
		# 	stock['recdate']=''
		# 	stock['To Grade']=''
		# 	stock['firm']=''
			
		# 	#stock=stock.append(pd.Series([s.ticker, s.cp, s.price, s.last_trade,s.change], index=stock.columns ), ignore_index=True)

		# 	for i in tqdm(stock_list['Ticker']):
		# 	    try:
		# 	        s=Stock(str(i))

		# 	        tickerData = yf.Ticker(i)
		# 	        stock=stock.append(pd.Series([i,s.cp, s.price, s.last_trade,s.change,tickerData.recommendations.index[-1],
		# 	                                  tickerData.recommendations.iloc[-1][1]
		# 	                                  ,tickerData.recommendations.iloc[-1][0]], index=stock.columns ), ignore_index=True)
		# 	    except:
		# 	        stock=stock.append(pd.Series([i,s.cp, s.price, s.last_trade,s.change, np.nan,np.nan,np.nan], index=stock.columns ), ignore_index=True)

		# 	st.balloons()
		# 	st.dataframe(stock)

		x = stock_list['Ticker'].to_string(header=False,
              index=False).split('\n')
		vals = [','.join(ele.split()) for ele in x]
		finwiz_url = 'https://finviz.com/quote.ashx?t='
		news_tables = {}
		#vals = ['CCL']

		for ticker in tqdm(vals):
		  try:
		    url = finwiz_url + ticker
		    req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
		    response = urlopen(req)    
		    # Read the contents of the file into 'html'
		    html = BeautifulSoup(response)
		    # Find 'news-table' in the Soup and load it into 'news_table'
		    news_table = html.find(id='news-table')
		    # Add the table to our dictionary
		    news_tables[ticker] = news_table
		  except Exception as e:
		    print (ticker, 'not found')
		st.balloons()
		parsed_news = []

		# Iterate through the news
		for file_name, news_table in news_tables.items():
		    # Iterate through all tr tags in 'news_table'
		    for x in news_table.findAll('tr'):
		        # read the text from each tr tag into text
		        # get text from a only
		        text = x.a.get_text() 
		        # splite text in the td tag into a list 
		        date_scrape = x.td.text.split()
		        # if the length of 'date_scrape' is 1, load 'time' as the only element

		        if len(date_scrape) == 1:
		            time = date_scrape[0]
		            
		        # else load 'date' as the 1st element and 'time' as the second    
		        else:
		            date = date_scrape[0]
		            time = date_scrape[1]
		        # Extract the ticker from the file name, get the string up to the 1st '_'  
		        ticker = file_name.split('_')[0]
		        
		        # Append ticker, date, time and headline as a list to the 'parsed_news' list
		        parsed_news.append([ticker, date, time, text])
		st.write('News Scraped')    
		st.write(parsed_news)
		st.write('Sentiment Analysis')
		import nltk
		nltk.downloader.download('vader_lexicon')
		# Instantiate the sentiment intensity analyzer
		vader = SentimentIntensityAnalyzer()

		# Set column names
		columns = ['ticker', 'date', 'time', 'headline']

		# Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
		parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

		# Iterate through the headlines and get the polarity scores using vader
		scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()

		# Convert the 'scores' list of dicts into a DataFrame
		scores_df = pd.DataFrame(scores)

		# Join the DataFrames of the news and the list of dicts
		parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

		# Convert the date column from string to datetime
		parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date

		st.dataframe(parsed_and_scored_news)

		mean_scores = parsed_and_scored_news.groupby(['ticker','date']).mean()

		# Unstack the column ticker
		mean_scores = mean_scores.unstack()

		# Get the cross-section of compound in the 'columns' axis
		mean_scores = mean_scores.xs('compound', axis="columns").transpose()
		st.write("Sum of Sentiment Daily")
		st.dataframe(mean_scores)

		abc=mean_scores.iloc[-no_days:].sum()>threshold
		pos_stocks=abc[abc].index.values
		st.write('Selected Stocks having sentiment greater than threshold')
		st.write(pos_stocks)

		# functions to get and parse data from FinViz

		def fundamental_metric(soup, metric):
		    return soup.find(text = metric).find_next(class_='snapshot-td2').text

		def get_fundamental_data(df):
		    for symbol in tqdm(df.index):
		        try:
		            url = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
		            headers = {'user-agent': 'my-app/0.0.1'}
		            soup = bs(requests.get(url, headers=headers).content, 'html.parser') 
		            for m in df.columns:                
		                df.loc[symbol,m] = fundamental_metric(soup,m)                
		        except Exception as e:
		            print (symbol, 'not found')
		    return df

		metric = ['Change','Price','Prev Close', 'Volume','Shs Outstand','Market Cap','P/B',
						'P/E',
						'Forward P/E',
						'PEG',
						'Debt/Eq',
						'EPS (ttm)',
						'Dividend %',
						'ROE',
						'ROI',
						'EPS Q/Q',
						'Insider Own','Inst Own']
		news_trend_tickers= list(pos_stocks)
		df = pd.DataFrame(index=news_trend_tickers,columns=metric)
		df = get_fundamental_data(df)
		st.balloons()
		df['sentiment']=mean_scores[pos_stocks].iloc[-no_days:].sum().T
		st.write('Stocks data of Sentiment greater than **{}**'.format(threshold))
		st.dataframe(df)
		def convert_scale(value):
		    if value.endswith("M"):
		        return float(value[:-1]) * 10**6
		    elif value.endswith("B"):
		        return float(value[:-1]) * 10**9
		    else:
		        return float(value)

		df["Shs Outstand"] = df["Shs Outstand"].apply(convert_scale)
		df["Market Cap"] = df["Market Cap"].apply(convert_scale)
		df['fair_sharevalue']= df["Market Cap"]/df["Shs Outstand"]
		df['Change']=df['Change'].str.replace('%', '')
		#df['Price']=df['Price']
		#df['Prev Close']=df['Prev Close']
		df['Volume']=df['Volume'].str.replace(',', '')
		df['Dividend %'] = df['Dividend %'].str.replace('%', '')
		df['ROE'] = df['ROE'].str.replace('%', '')
		df['ROI'] = df['ROI'].str.replace('%', '')
		df['EPS Q/Q'] = df['EPS Q/Q'].str.replace('%', '')
		df['Insider Own'] = df['Insider Own'].str.replace('%', '')
		df['Inst Own'] = df['Inst Own'].str.replace('%', '')
		df = df.apply(pd.to_numeric, errors='coerce')
		st.write('After Data Wrangling')
		st.dataframe(df)
		def to_excel(df):
		    output = BytesIO()
		    writer = pd.ExcelWriter(output, engine='xlsxwriter')
		    df.to_excel(writer, index = False, sheet_name='Sheet1',float_format="%.2f")
		    writer.save()
		    processed_data = output.getvalue()
		    return processed_data

		def get_table_download_link(df):
		    """Generates a link allowing the data in a given panda dataframe to be downloaded
		    in:  dataframe
		    out: href string
		    """
		    val = to_excel(df)
		    b64 = base64.b64encode(val)  # val looks like b'...'
		    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Your_File.xlsx">Download file</a>' # decode b'abc' => abc
		st.write('**Download the Positive Sentiment Stocks**')
		df.reset_index(level=df.index.names, inplace=True)
		st.markdown(get_table_download_link(df), unsafe_allow_html=True)
		



if __name__ == '__main__':
	main()
