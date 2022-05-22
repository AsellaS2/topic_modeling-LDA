#-*- coding: utf-8 -*-

from kiwipiepy import Kiwi
import tomotopy as tp
import pandas as pd


# 전처리 과정 (+불용어처리)

kiwi = Kiwi()
kiwi.prepare()
stopwords = set(["사람", "것"])

# tokenize 함수를 정의함. 한국어 문장을 입력하면 형태소 단위로 분리하고, 불용어 및 특수 문자 등을 제거한 뒤 list로 반환
def tokenize(sent):
    res, score = kiwi.analyze(sent)[0] # 첫번째 결과를 사용
    return [word + ('다' if tag.startswith('V') else '') # 동사에는 '다'를 붙여줌
            for word, tag, _, _ in res
            if not tag.startswith('E') and not tag.startswith('J') and not tag.startswith('S') and word not in stopwords] # 조사, 어미, 특수기호 및 stopwords에 포함된 단어는 제거




# LDAModel을 생성
# 토픽의 개수(k)는 20개, alpha 파라미터는 0.1, eta 파라미터는 0.01
# 전체 말뭉치에 5회 미만 등장한 단어들은 제거

model = tp.LDAModel(k=20, alpha=0.1, eta=0.01, min_cf=5)


# input_file.txt 파일에서 한 줄씩 읽어와서 model에 추가

# for i, line in enumerate(open('C:/Users/YSS/Desktop/input_file.txt', encoding='utf-8')):
for i, line in enumerate(open('C:/Users/YSS/Desktop/input_file.txt')):
    # model.add_doc(line.strip().split()) # 공백 기준으로 단어를 나누어 model에 추가함
    model.add_doc(tokenize(line)) # tokenize함수를 이용해 전처리한 결과를 add_doc에 넣어줌
    if i % 10 == 0: print('Document #{} has been loaded'.format(i))


# model의 num_words나 num_vocabs 등은 train을 시작해야 확정됨
# 따라서 이 값을 확인하기 위해서 train(0)을 하여 실제 train은 하지 않고 학습 준비만 함
# num_words, num_vocabs에 관심 없다면 이부분은 생략가능

model.train(0)
print('Total docs:', len(model.docs))
print('Total words:', model.num_words)
print('Vocab size:', model.num_vocabs)


# 다음은 train을 총 200회 반복하면서, 매 단계별로 로그 가능도 값을 출력함
# 혹은 단순히 model.train(200)으로 200회 반복도 가능

for i in range(200):
    print('Iteration {}\tLL per word: {}'.format(i, model.ll_per_word))
    model.train(1)


result = pd.DataFrame()

# 학습된 토픽들을 출력

for i in range(model.k):
    # 토픽 개수가 총 20개이니, 0~19번까지의 토픽별 상위 단어 10개를 뽑아봅시다.
    res = model.get_topic_words(i, top_n=10)
    topic_num = 'Topic #{}'.format(i)
    rlist = list(w for w, p in res)
    df1 = pd.DataFrame(rlist, columns={topic_num})
    result = pd.concat([result,df1], axis=1)

print(result)

# csv로 저장
result.to_csv("C:/Users/YSS/Desktop/result.csv", mode='w', encoding='utf-8-sig')

