#!/usr/bin/env python3
import time
import datetime
from urllib.request import urlopen
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd


class ReutersCrawler(object):
    def __init__(self):
        self.ticker_list = open('./input/tickerList.csv')

        self.finished_tickers = set()       #let it be but currently of no use to the project since the domain has been restricted
        try: # this is used when we restart a task
            finished = open('./input/finished.reuters')
            for ticker in finished:
                self.finished_tickers.add(ticker.strip())
        except:
            pass

    def fetch_content(self, ticker, name, line, date_list, exchange):
        
        suffix = {'National Stock Exchange': '.NS'}
        
        url = "http://www.reuters.com/finance/stocks/company-news/" + ticker + suffix[exchange]
        print(url)
        has_content = 0
        repeat_times = 4
        

        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        has_content = len(soup.find_all("div", {'class': ['topStory', 'feature']}))
        print(has_content)    
        # spider task for the past
        # if some company has no news even if we don't input date
        #     set this ticker into the lowest priority list
        # else
        #     if it doesn't have a single news for NN consecutive days, stop iterating dates
        #     set this ticker into the second-lowest priority list


        ticker_failed = open('./input/news_failed_tickers.csv', 'a+')
        if has_content > 0:
            missing_days = 0

            #print(date_list)
            for timestamp in date_list:

                #print(timestamp)
                has_news = self.repeat_download(ticker, line, url, timestamp)
                #print("has_news"+str(has_news))    

                if has_news:
                    missing_days = 0 # if get news, reset missing_days as 0
                else:
                    missing_days += 1
                
                # 2 NEWS: wait 30 days and stop, 10 news, wait 70 days
                if missing_days > has_content * 5 + 20:
                    break # no news in X consecutive days, stop crawling

                if missing_days > 0 and missing_days % 20 == 0:
                    print("%s has no news for %d days, stop this candidate ..." % (ticker, missing_days))
                    ticker_failed.write(ticker + ',' + timestamp + ',' + 'LOW\n')
        else:
            print("%s has no news" % (ticker))
            today = datetime.datetime.today().strftime("%Y%m%d")
            ticker_failed.write(ticker + ',' + today + ',' + 'LOWEST\n')
        ticker_failed.close()

    def repeat_download(self, ticker, line, url, timestamp):    
        new_time = timestamp[4:] + timestamp[:4] # change 20151231 to 12312015 to match reuters format

        #print(url + "?date=" + new_time)
        page = requests.get(url + "?date=" + new_time)
        soup = BeautifulSoup(page.content, 'html.parser')
        has_news = self.parser(soup, line, ticker, timestamp)
        print(url + "?date=" + new_time+"     "+str(has_news))

        if has_news:
            return 1

        return 0                #repeat download done (not downloading many time coz it was giving some error)



    def parser(self, soup, line, ticker, timestamp):
        content = soup.find_all("div", {'class': ['topStory', 'feature']})
        if not content:
            return 0
        fout = open('./input/news_reuters.csv', 'a+')
        for i in range(len(content)):
            title = content[i].h2.get_text().replace(",", " ").replace("\n", " ")
            body = content[i].p.get_text().replace(",", " ").replace("\n", " ")

            if i == 0 and soup.find_all("div", class_="topStory"):
                news_type = 'topStory'
            else:
                news_type = 'normal'

            print(ticker, timestamp, title, news_type)
            fout.write(str(','.join([timestamp, title, body, news_type]).encode('utf-8') ))
            fout.write('\n')
        fout.close()
        return 1

    def generate_past_n_day(self, numdays):
        
        """Generate N days until now"""
        #base = datetime.datetime.today()
        #date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
        
        start_date = datetime.date(2009, 1, 1)
        end_date   = datetime.date(2011, 5, 4) #make the necessary changes in the date to update the data

        date_list = [ start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days))]
        date_list.reverse()    

        #date_list = pd.date_range(pd.datetime.today(), periods=numdays).tolist()
        #print(date_list)
        return [x.strftime("%Y%m%d") for x in date_list]

    def run(self, numdays=50):
        """Start crawler back to numdays"""
        date_list = self.generate_past_n_day(numdays) # look back on the past X days
        for line in self.ticker_list: # iterate all possible tickers
            line = line.strip().split(',')
            ticker, name, exchange, market_cap = line
            if ticker in self.finished_tickers:
                continue
            print("%s - %s - %s - %s" % (ticker, name, exchange, market_cap))
            self.fetch_content(ticker, name, line, date_list, exchange)


def main():
    reuter_crawler = ReutersCrawler()
    reuter_crawler.run(1)

if __name__ == "__main__":
    main()
