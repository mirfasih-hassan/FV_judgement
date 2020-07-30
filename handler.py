try:
  import unzip_requirements
except ImportError:
  pass

import os
import logging
import requests
from google.cloud import vision
from google.cloud.vision import types
import numpy as np
from numpy import array
from sklearn.metrics.pairwise import euclidean_distances

# location of the JSON file of Google Application Credentials to use Google API

logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './gcp.json'

def hello(event, context):
    logger.info('### EVENT')
    logger.info(event)

    body = event.get('body')

    logger.info('### BODY')
    logger.info(body)
    
    url = body.get('url')
    
    response = requests.get(url)
    image = response.content
    client = vision.ImageAnnotatorClient()


    ############# Detect Text by GOOGLE API #############################
    
    content_image = types.Image(content=image)
    response = client.document_text_detection(image = content_image)
    texts = [text.description for text in response.text_annotations]
    text = texts[0]
    print(text)

    ############ Detect dominant colors by GOOGLE API ###################

    image = vision.types.Image(content=image)
    response = client.image_properties(image=image).image_properties_annotation
    dominant_colors= response.dominant_colors
    print(dominant_colors)

    fwl = ['キレイ','しっとり輝く', 'ホワイトニング', '美肌', '肌がきれいになる', '赤ちゃんのような',
      'デリケートゾーン', 'すっぴん', '健康的なお肌', 'エイジングケア', 'ハリコシ実感',
      'ダイエット', '刃リハリ', '手汗', '竹炭', 'インナーケア', 'スキニーが履けない',
      'レギンス', '美脚&美尻&美腹', 'ヒップアップ', '骨盤矯正', 'バストケア',
      'ジュエルアップ', '満足の声', 'ボディケアサプリ', 'キレイになって', 'カロリーサポート',
      'レディース', '女性', '栄養補助食品', 'いちご味', 'ふたえ美容液',
      'おしりの黒ずみがきになって', '美人', 'デリケート', '美容エキス',
      'もっと若々しく見られたい', '化粧のりUP.', '話題の新習慣で若見え',
      'ふわっ', 'スキンケア', '真珠エキス', '美容保湿成分配合', 'バストケアクリームジェル部門',
      '女性がオススメしたい', '美ボディへ', '美ボディ期待度', '年相応に若々しくいたい',
      '健康的な体をつくる', 'シャルーヌ化粧品', '妊娠', 'ポロリケア', '驚きの満足度',
      'ハニプラ石鹸', 'セクシーヒップ!!', 'スッキリ肌', '洗浄ケア', '保湿ケア',
      '美白クリーム', '体内フローラ', '美容ボタニカル', '植物成分', '美人通販',
      '若々しさの秘密', '真実の一滴をあなたの肌に', '至極の潤い', '100%美容原液',
      '鮮度の高い', 'からだイキイキ', '首ケア化粧品ロコミ評', 'シルキースワン',
      'ご使用になられたお客様より', '薬用育毛剤', 'ぱっちり輝かしい目元に',
      '無菌獎蛋', 'コルジセピン', 'keratin', '保湿', '高純度', '女性誌', '潤い、ハリ肌!',
      '美白ケア', 'シンデレラ', '母乳', '美容液', '肌老化', '先輩ママ', '健康を育む',
      '美習慣!', '気持ちの前向きサプリ', 'その髪、水素で美しく', '敏感肌用ソープ',
      '満足度', 'インナーケア']
    
    # male dictionary of words
    mwl = ['男肌', 'デキる男', 'お通じ改善', 'キャビキシル', 'ピディオキシジル', 'ロ臭',
       '筋肉をデザイン', '親子満足度', ' プロテオグリカン', '頭皮', '满足','エネルギー']


    Farr = array([[4.49078023e-01, 3.52895796e-01, 2.49000000e+02, 2.45000000e+02,
        2.45000000e+02],
       [2.69503538e-02, 2.10211530e-01, 2.27000000e+02, 2.80000000e+01,
        3.40000000e+01],
       [5.14893606e-02, 5.94812073e-02, 2.31000000e+02, 1.88000000e+02,
        1.95000000e+02],
       [2.83687934e-03, 5.61073795e-03, 2.47000000e+02, 1.29000000e+02,
        2.10000000e+01],
       [2.12765951e-03, 4.63203946e-03, 7.80000000e+01, 4.10000000e+01,
        2.00000000e+01],
       [3.54609918e-03, 3.95824946e-03, 1.24000000e+02, 7.40000000e+01,
        3.30000000e+01],
       [1.71631202e-02, 7.19830841e-02, 2.26000000e+02, 3.90000000e+01,
        6.10000000e+01],
       [4.37588654e-02, 5.40004633e-02, 2.53000000e+02, 2.19000000e+02,
        2.27000000e+02],
       [1.84397157e-02, 2.84885298e-02, 2.27000000e+02, 4.90000000e+01,
        9.10000000e+01],
       [1.84397157e-02, 2.12162342e-02, 2.46000000e+02, 2.31000000e+02,
        2.11000000e+02]])
    
    Marr = array([[4.76439387e-01, 1.80112511e-01, 1.70000000e+01, 1.40000000e+01,
        1.40000000e+01],
       [4.16666688e-03, 9.00585130e-02, 2.20000000e+02, 1.84000000e+02,
        3.80000000e+01],
       [7.34848483e-03, 3.74001600e-02, 1.38000000e+02, 1.19000000e+02,
        6.10000000e+01],
       [6.36363635e-03, 2.86419187e-02, 2.15000000e+02, 1.47000000e+02,
        1.60000000e+01],
       [6.51515136e-03, 2.38234107e-03, 1.81000000e+02, 6.80000000e+01,
        8.30000000e+01],
       [9.61363614e-02, 5.57572283e-02, 5.20000000e+01, 5.00000000e+01,
        5.20000000e+01],
       [7.52272755e-02, 4.31090891e-02, 8.50000000e+01, 8.20000000e+01,
        8.50000000e+01],
       [6.06060587e-03, 3.87626924e-02, 2.19000000e+02, 1.86000000e+02,
        9.10000000e+01],
       [5.90909086e-03, 3.55742164e-02, 1.82000000e+02, 1.52000000e+02,
        6.00000000e+01],
       [7.19696982e-03, 3.15366350e-02, 1.00000000e+02, 8.30000000e+01,
        2.80000000e+01]])


    if any(fw in text for fw in fwl):
        print("any........................")
        # Save dominant coloes information as array
        PF = [] # pixel fraction
        SV = [] # score value
        R = []  # Red
        G = []  # Green
        B = []  # Blue
        for color in dominant_colors.colors:
            PF.append(color.pixel_fraction)
            SV.append(color.score)
            R.append(color.color.red)
            G.append(color.color.green)
            B.append(color.color.blue)
        arr = np.concatenate((np.transpose(np.array(PF).reshape(-1,1)),
                              np.transpose(np.array(SV).reshape(-1,1)),
                              np.transpose(np.array(R).reshape(-1,1)),
                              np.transpose(np.array(G).reshape(-1,1)),
                              np.transpose(np.array(B).reshape(-1,1))), axis =0)
        Tarr = np.transpose(arr)
        ED = euclidean_distances(Tarr, Farr).mean()
        z = abs(ED-160.8980699302674)/(171.84327257462073-160.8980699302674)
        print(z)
        if z < 1.5:
            result = 'For Woman'
        else:
            result = 'For man'

    elif any(mw in text for mw in mwl):
        print('MW')

        PF = [] # pixel fraction
        SV = [] # score value
        R = []  # Red
        G = []  # Green
        B = []  # Blue
        for color in dominant_colors.colors:
            PF.append(color.pixel_fraction)
            SV.append(color.score)
            R.append(color.color.red)
            G.append(color.color.green)
            B.append(color.color.blue)
        arr = np.concatenate((np.transpose(np.array(PF).reshape(-1,1)),
                              np.transpose(np.array(SV).reshape(-1,1)),
                              np.transpose(np.array(R).reshape(-1,1)),
                              np.transpose(np.array(G).reshape(-1,1)),
                              np.transpose(np.array(B).reshape(-1,1))), axis =0)
        Tarr = np.transpose(arr)
        ED = euclidean_distances(Tarr, Marr).mean()
        z = (ED-160.8980699302674)/(171.84327257462073-160.8980699302674)
        if z < 1.55:
            result = 'For man'
        else:
            result = 'For Woman' 
    else:
        result = 'Else'

                

    response = {
        'statusCode': 200,
        'data': {
          'result': result,
          'url': url
        }
    }
    print(response)
    return response
