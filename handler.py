import io
import os
import logging
import requests
import json
from google.cloud import vision
from google.cloud.vision import types
from google.cloud import storage
from google.oauth2 import service_account
import numpy as np
from numpy import array
from sklearn.metrics.pairwise import euclidean_distances




logger = logging.getLogger()
logger.setLevel(logging.INFO)


# location of the JSON file of Google Application Credentials to use Google API
#gac_loc = '/mnt/raid1/hassan/chalklet-develop-31f393ccb359.json'

#os.environ['GOOGLE_APPLICATION_CREDENTIALS']= gac_loc


def hello(event, context):
    logger.info('### EVENT')
    logger.info(event)

    body = event.get('body')

    logger.info('### BODY')
    logger.info(body)

#    url = body.get('url')
    url = "http://ahjikan-shop.com/lp_nd/goboutea/images/key_benpi2.jpg"

    logger.info('### URL')
    logger.info(url)
        
    # image URL
    # ex.
    # http://ahjikan-shop.com/lp_nd/goboutea/images/key_benpi2.jpg

    # judge logic:        
    # (if needed) download image file
    # Get the URL of the FV and download and save it
    
    file_name = "testFV.jpg"    
    response = requests.get(url)
    image = response.content
    client = vision.ImageAnnotatorClient()

    with open(file_name, "wb") as aaa:
       aaa.write(image)

    # Read the saved image from URL
    # fvimg = cv2.imread("testFV.jpg")    

    # access google API
    # import the credentials file to make use of API services

    with open('/mnt/raid1/hassan/chalklet-develop-31f393ccb359.json') as source:
       info = json.load(source)

    storage_credentials = service.account.Credentials.from_service_account_info(info)

    storage_client = storage.Client(project=project_id, credentials=storage_credentials)
    
    
    # import the Google Cloud client library
    #from google.cloud import storage
    
    # Instantiates a client
    #storage_client = storage.Client()
    
    # The name for the new bucket
    #bucket = storage_client.create_bucket(bucket_name)

    #print("Bucket {} created.".format(bucket.name))



    ############# Detect Text by GOOGLE API #############################
    with io.open(file_name, 'rb') as image_file1:
        content = image_file1.read()
    content_image = types.Image(content=content)
    response = client.document_text_detection(image = content_image)
    texts = [text.description for text in response.text_annotations]
    text = texts[0]
    
    ############ Detect dominant colors by GOOGLE API ###################
    
    image = vision.types.Image(content=content)
    response = client.image_properties(image=image).image_properties_annotation    
    dominant_colors= response.dominant_colors
    
    
    fwl = ['しっとり輝く', 'ホワイトニング', '美肌', '肌がきれいになる', '赤ちゃんのような',
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
      '満足度']
   
    
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
    
    
    if any(fw in text for fw in fwl):
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
        z = (ED-160.8980699302674)/(171.84327257462073-160.8980699302674)
             
        if z < 0.5:
            result = 'For Woman'            
            
        else:
            result = 'For man'                           


    response = {
        'statusCode': 200,
        'data': {
          'result': result
        }
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """
