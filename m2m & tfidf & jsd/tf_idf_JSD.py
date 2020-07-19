from newspaper import Article
import urllib.request
from bs4 import BeautifulSoup
import ssl
import datetime
import re
import pandas as pd

from konlpy.tag import Kkma, Twitter, Komoran
from sklearn.feature_extraction.text import TfidfVectorizer
from math import exp

context = ssl._create_unverified_context()




def list_crawling(date_list):'''주어진 날짜에 대해, 각 날짜별로 천여개의 스크랩할 주소리스트 저장'''

    url_1='https://media.daum.net/breakingnews/economic?regDate='
    url_2='&page='
    page= range(1,10)

    total_url=[]

    for date in date_list:
        k=0
        print("크롤링할 url 주소 생성중.. %d %%"%(100* (k/len(date_list))))
        k=k+1
        url_list=[]
        for i in range(date,date+7):
            for j in page:
                mainurl=url_1 +str(i)+ url_2 + str(j)
                source_code_from_URL = urllib.request.urlopen(mainurl, context=context)
                soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
                list_news = soup.find('ul', attrs={'class':'list_news2'})
                list_news_li = list_news.find_all('li')
                for item in list_news_li:
                    link_txt = item.find('a', attrs={'class':'link_txt'})
                    url_list.append(link_txt.get('href'))
        total_url.append(url_list)

    return total_url

'''저장한 주소리스트에 대해 크롤링을 진행하고, 파싱을 진행한 후 명사만을 추출하여 tf-idf 모델에 적용시켰습니다. : 24개의 날들에 대해 기사들에서 나타나는 단어들의 분포를, idf값의 역수인 df값으로 나타내기 위함'''
def news_crawling(total_url):

    total_tfidf=[]

    for url_list in total_url:
        kkma = Kkma()
        mydoclist_kkma = []
        k = 0
        print("뉴스 크롤링중.. %d %%" % (100 * (k / len(total_url))))
        k = k + 1
        for url in url_list:
            sleep(0.05)
            article = Article(url, language='ko')
            article.download()
            article.parse()
            hoho = article.title + article.text

            kkma_nouns = ' '.join(kkma.nouns(hoho))
            mydoclist_kkma.append(kkma_nouns)

        tfidf_vectorizer = TfidfVectorizer(min_df=1)
        tfidf_vectorizer.fit(mydoclist_kkma)
        total_tfidf.append(tfidf_vectorizer)
    return total_tfidf

'''tf-idf를 적용하여 얻어낸 결과에서 단어별로 idf값을 먼저 저장합니다'''
def doc_term_idf(total_tfidf):
    dist_idf=[] #문서별, 단어들의 idf 값 분포 리스트

    for i in total_tfidf:
        term_idf_dict={}
        for j in range(len(list(i.vocabulary_.keys()))):
            term_idf_dict[list(i.vocabulary_.keys())[j]]=i.idf_[j]
        dist_idf.append(term_idf_dict)
    return dist_idf

#KL Divergence를 계산해주는 함수
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
# idf 값을 인풋으로 넣고, df값으로 역산하여 그 분포들의 유사도 jsd값을 구하는 함수
'''비교하고자 하는 두 날의 idf 값 분포를 인풋으로 넣으면, df값으로 역산하고 두 날의 분포 유사도를 Jensen-Shannon Divergence로 산출해주는 함수를 정의'''
def jsd(date1,date2):
    date1_term_dict={}
    date2_term_dict={}

    #먼저 비교하고자 하는 두 날의 기사의 단어들이 다르기 때문에 단어들을 먼저 합쳐주는 작업을 진행 : JSD값을 얻어내기 위해 벡터길이를 맞춰야하기 때문에 단어들의 합집합으로 만들어줌
    for i in date1.keys():
        date1_term_dict[i]=0
        date2_term_dict[i]=0
    for j in date2.keys():
        date1_term_dict[j]=0
        date2_term_dict[j]=0

    #합쳐준 뒤, 원래 벡터값들을 각각 살려서 저장
    for i in date1.keys():
        date1_term_dict[i]=date1[i]
    for j in date2.keys():
        date2_term_dict[j]=date2[j]
    date1_idf=list(date1_term_dict.values())
    date2_idf=list(date2_term_dict.values())


    # 기본 tf-idf 함수에서 제공하는 식을 기준으로 df값을 역산하는 과정
    for i in range(len(date1_idf)):
        if date1_idf[i]!=0:
            date1_idf[i]=1/exp(date1_idf[i]-1)
        if date2_idf[i]!=0:
            date2_idf[i]=1/exp(date2_idf[i]-1)

    # df값을 전체 합으로 나눔으로써 단어 확률들의 합을 1로 만들어주는 과정
    dfsum_date1=sum(date1_idf)
    dfsum_date2=sum(date2_idf)
    date1_idf_final=date1_idf
    date2_idf_final=date2_idf
    for i in range(len(date1_idf_final)):
        date1_idf_final[i]=date1_idf[i]/dfsum_date1
        date2_idf_final[i]=date2_idf[i]/dfsum_date2

    M=[(i+j)/2 for i,j in zip(date1_idf_final, date2_idf_final)]

    return (KL(date1_idf_final, M) +KL(date2_idf_final, M))/2 # JSD 값을 리턴해주는 모습


if __name__ == "__main__":
    # many2many에서 학습된 결과와 랜덤하게 선정한 24개의 날짜 리스트 정의
    date_list = [20091120, 20110729, 20101011, 20101125, 20110204, 20110331, 20110211, 20110403, 20110309, 20110420,
                 20110511, 20110623, 20120721, 20130206, 20130825, 20140313, 20140929, 20150417, 20151103, 20160521,
                 20161207, 20170625, 20180111, 20180730]

    '''앞서 지정한 24개의 날짜에 대해서, 이전 일주일간의 스크랩할 주소리스트를 생성 / 7일 기준 * 9page * 15개 약 천여개의 기사 스크랩 '''
    total_url = list_crawling(date_list)
    total_tfidf = news_crawling(total_url)

    # 스크랩한 기사별로 단어와 idf값으로 이루어진 리스트 생성
    dist_idf = doc_term_idf(total_tfidf)

    ''' 24개의 날의 모든 경우(24*23 / 2 가지 경우의 수)에 대해 모두 유사도 측정하여 데이터프레임 형식으로 저장'''
    jsd_all=[]
    for i in range(len(dist_idf)):
        jsd_row=[]
        for j in range(len(dist_idf)):
            jsd_row.append(round(jsd(dist_idf[i],dist_idf[j]), 4))
        jsd_all.append(jsd_row)

    jsd_matrix=pd.DataFrame(jsd_all) #jsd_matrix는 위의 날짜 순서대로, 각 날짜별로 기사들의 단어 분포들이 가지는 유사도를 의미하는 매트릭스.
    jsd_matrix
