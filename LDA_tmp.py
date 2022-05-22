import pandas as pd
from pandas import DataFrame as df
import numpy as np
from collections import Counter
from konlpy.tag import Okt
okt = Okt()
import matplotlib.pyplot as plt
import re
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
from datetime import datetime
import os



Data = pd.read_csv('./DATA/Seoul_2015.csv', encoding='utf8')

Data['time'] = pd.to_datetime(Data["time"])

element_count = {}
for item in Data['time']:
    element_count.setdefault(item, 0)
    element_count[item] += 1
doc_count = pd.DataFrame.from_dict(element_count, orient='index', columns=["doc_count"])

Data.total = Data.total.astype(str)

clean_Data = Data

#데이터 프레임의 'text' 열의 값 중 keyword1이나 keyword 2가 포함된 행은 삭제
#clean_Data = Data[~Data['text'].str.contains('keyword1|keyword2')]

#text와 timestamp 열을 기준으로 중복된 데이터를 삭제, inplace : 데이터 프레임을 변경할지 선택(원본을 대체)
clean_Data.drop_duplicates(subset=['total','time'], inplace=True)

#한글이 아니면 빈 문자열로 바꾸기
clean_Data['total'] = clean_Data['total'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣]',' ',regex=True)

#빈 문자열 NAN 값으로 바꾸기
clean_Data = clean_Data.replace({'': np.nan})
clean_Data = clean_Data.replace(r'^\s*$', None, regex=True)

#NAN 이 있는 행은 삭제
clean_Data.dropna(how='any', inplace=True)

#인덱스 차곡차곡
clean_Data = clean_Data.reset_index(drop=True)



#텍스트 데이터를 리스트로 변환
Data_list = clean_Data.total.values.tolist()

Data_list[:100]

okt = Okt()
#리스트를 요소별로(트윗 하나) 가져와서 명사만 추출한 후 리스트로 저장
data_word=[]
for i in range(len(Data_list)):
    try:
        data_word.append(okt.nouns(Data_list[i]))
    except Exception as e:
        continue

print(data_word)

# id2word=corpora.Dictionary(data_word)
# id2word.filter_extremes(no_below = 20)
# texts = data_word
# corpus=[id2word.doc2bow(text) for text in texts]
#
# os.environ['MALLET_HOME'] = 'C:/Users/YSS/Downloads/mallet-2.0.8'
#
# mallet_path = 'C:/Users/YSS/Downloads/mallet-2.0.8/bin/mallet'
# ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word)
#
#
# coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
# coherence_ldamallet = coherence_model_ldamallet.get_coherence()
#
#
# def compute_coherence_values(dictionary, corpus, texts, limit, start=4, step=2):
#
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=data_word, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())
#
#     return model_list, coherence_values
#
#
#
# # Can take a long time to run.
# model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, start=4, limit=21, step=2)
#
# limit = 21;
# start = 4;
# step = 2;
# x = range(start, limit, step)
# topic_num = 0
# count = 0
# max_coherence = 0
# for m, cv in zip(x, coherence_values):
#     print("Num Topics =", m, " has Coherence Value of", cv)
#     coherence = cv
#     if coherence >= max_coherence:
#         max_coherence = coherence
#         topic_num = m
#         model_list_num = count
#     count = count + 1
#
# # Select the model and print the topics
# optimal_model = model_list[model_list_num]
# model_topics = optimal_model.show_topics(formatted=False)
#
#
# # print(optimal_model.print_topics(num_words=10))
#
#
# def format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=texts):
#     # Init output
#     sent_topics_df = pd.DataFrame()
#
#     # Get main topic in each document
#     # ldamodel[corpus]: lda_model에 corpus를 넣어 각 토픽 당 확률을 알 수 있음
#     for i, row in enumerate(ldamodel[corpus]):
#         row = sorted(row, key=lambda x: (x[1]), reverse=True)
#         # Get the Dominant topic, Perc Contribution and Keywords for each document
#         for j, (topic_num, prop_topic) in enumerate(row):
#             if j == 0:  # => dominant topic
#                 wp = ldamodel.show_topic(topic_num, topn=10)
#                 topic_keywords = ", ".join([word for word, prop in wp])
#                 sent_topics_df = sent_topics_df.append(
#                     pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
#             else:
#                 break
#     sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
#     print(type(sent_topics_df))
#
#     # Add original text to the end of the output
#     # contents = pd.Series(texts)
#     # sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
#     sent_topics_df = pd.concat(
#         [sent_topics_df, Data['text'], Data['timestamp'], Data['tweet_url'], Data['screen_name'], Data['label'],
#          Data['clean']], axis=1)
#     return (sent_topics_df)
#
#
# df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=Data_list)
#
# # Format
# df_topic_tweet = df_topic_sents_keywords.reset_index()
# df_topic_tweet.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'Timestamp',
#                           'Tweet_url', 'Screen_name', 'label', 'Clean']
#
# # Show각 문서에 대한 토픽
# # df_dominant_topic=df_dominant_topic.sort_values(by=['Dominant_Topic'])
# # df_topic_tweet
#
#
# # Group top 5 sentences under each topic
# sent_topics_sorteddf_mallet = pd.DataFrame()
#
# sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
#
# for i, grp in sent_topics_outdf_grpd:
#     sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
#                                              grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
#                                             axis=0)
#
# # Reset Index
# sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
#
# topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
# topic_counts.sort_index(inplace=True)
#
# topic_contribution = round(topic_counts / topic_counts.sum(), 4)
# topic_contribution
#
# lda_inform = pd.concat([sent_topics_sorteddf_mallet, topic_counts, topic_contribution], axis=1)
# lda_inform.columns = ["Topic_Num", "Topic_Perc_Contrib", "Keywords", "Text", "timestamp", "Clean", "tweet_url",
#                       "screen_name", "label", "Num_Documents", "Perc_Documents"]
# lda_inform = lda_inform[["Topic_Num", "Keywords", "Num_Documents", "Perc_Documents"]]
# lda_inform
# # lda_inform.Topic_Num = lda_inform.Topic_Num.astype(int)
# lda_inform['Topic_Num'] = lda_inform['Topic_Num'] + 1
# lda_inform.Topic_Num = lda_inform.Topic_Num.astype(str)
# lda_inform['Topic_Num'] = lda_inform['Topic_Num'].str.split('.').str[0]
# df_topic_tweet['Dominant_Topic'] = df_topic_tweet['Dominant_Topic'] + 1
# df_topic_tweet.Dominant_Topic = df_topic_tweet.Dominant_Topic.astype(str)
# df_topic_tweet['Dominant_Topic'] = df_topic_tweet['Dominant_Topic'].str.split('.').str[0]
#
# lda_inform.to_csv ("./Result/lda_inform.csv", index = None)
# print(lda_inform)