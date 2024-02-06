#!/usr/bin/env python
# coding: utf-8

if(__name__ == '__main__'):
# In[1]:
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import keras
    import os
    import mysql.connector
    from sklearn.metrics import accuracy_score
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, GlobalMaxPooling1D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize



    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)


    # In[ ]:





    # In[2]:


    from keras.models import load_model

    os.chdir(r'C:\xampp\htdocs\Virginitea\output\Final_Sentiment_Analysis')
    new_model=keras.models.load_model('may_20/')
    mtea_df=pd.read_csv('Main_Dataset.csv')
    print("Model Loaded")

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(mtea_df['Text'].values, mtea_df['Rating'].values, test_size=0.20,random_state=42)


    # In[6]:


    #tokenizing
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    max_vocab=2000000
    tokenizer= Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(x_train)


    # In[7]:


    word_indx=tokenizer.word_index
    V=len(word_indx)


    # In[8]:


    train_seq = tokenizer.texts_to_sequences(x_train)
    test_seq = tokenizer.texts_to_sequences(x_test)


    # In[9]:


    #padding
    pad_train = pad_sequences(train_seq,maxlen=547)
    T=pad_train.shape[1]


    # In[10]:


    pad_test = pad_sequences(test_seq, maxlen=547)


    # In[ ]:





    # In[15]:


    #sentiment detecting
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    def predict_sentiment(text):
        #preprocessing
        text_seq = tokenizer.texts_to_sequences(text)
        text_pad = pad_sequences(text_seq, maxlen=T)

        #predicting the class
        predicted_sentiment = new_model.predict(text_pad)
        #rounded = [np.round(text,2) for x in predicted_sentiment]
        rounded_predicted_sentiment = np.round(predicted_sentiment,2)
        good = "'Good Sentiment'"
        bad = "'Bad Sentiment'"

        if rounded_predicted_sentiment.round() == 1.0:
            return print("Good Sentiment")

        else:
            return print("Bad Sentiment")




    #bad
    #text=["This doesn't look freshly made. Really disappointed! All the tapioca was stuck together, impossible to drink with a straw. Drink didn't taste fresh."]
    #text=["If this was my first bubble milk tea, I would not buy or crave for more. If you want to pay £8 for two water milk tea, please try it here. Real low & cheap quality. Never again!!!!!"]
    #text=["This place is pretty terrible, I don't know if you can call it a bubble tea store when it serves things that taste like water"]
    #text=["It doesn't taste refreshing, i would rather throw it into the trash than drink it"]
    #text=['It is too sweet to my liking']
    #text=["I ordered 2 oolong milk teas, it was very diluted tea and tasted like water. Ice was melting into the drink. I ordered nondairy creamer in one and they did not specify which one had it. Horrible taste, awful quality. :("]
    text=["Not real milk/bubble tea. The first thing I see is that the menu either says Smoothies or Over Ice for the classic flavors. And then I asked the cashier if they have milk tea, and they said they could do any of the flavors on that list because they're made from powder. Yes, you heard me, all of their flavors are made from powder, even their so-called teas!!! I was so very disappointed, and I ordered the thai tea because I have tried instances where that powder base has been alright. I was proven wrong. It was quite honestly some of the worst milk tea I have ever had. I might as well have called it milk with a spoonful of thai tea powder. Call me spoiled coming from California, but that simply doesn't cut it for me. Will be looking to try milk tea in Chinatown. My cousin also was not a fan of his Oreo smoothie from here either...we honestly tried this place hoping that it would be somewhat decent based off of Yelp reviews. Sadness."]
    #text=["I ordered 2 drinks and they mixed up the toppings, which you know, completely ruins the drinks for the customer. Additionally, the red bean was sour - meaning it has gone bad. I immediately got a stomach ache from ingesting one sip. I feel nauseous. Trying to call the store to tell them to stop using their red bean, but no one is answering and now it sounds like they have taken the phone off the hook as there is a busy signal for a long time."]


    #good
    #text=["it taste good"]
    #text=["It is very tasty I love it so much"]
    #text=["This place is really cute! There's cool art on the wall and the staff are nice. We got our drinks pretty quickly and they all tasted really good. My friend got a strawberry slushy and it came with a super cute cup. The only problem is that it's kind of hard to find. My friends and I had to make it to the rogers center for a certain time and this place was the closest, so we looked for it for a good 15 minutes before we found it. It's inside a building with no real indication it's in there. Other than that, great place!"]
    #text=["I mostly stick to fresh juices and shakes but there are times when I compromise my inherent need for anything healthy- this is when you'll find me at Happy Lemon. I only order one thing though and it's the Cocoa with Rock Salt and Cheese. I like the contrast between the salt and the sweet. It's very filling though and I can't finish a cup all by myself but still it's a drink worthy of your P90. This branch in High Street is always filled with Happy Lemon fans-yuppies, students, ladies-who-lunch they love it here and it's all within reason. Drinks here are prepared efficiently,you never have to wait long. Also, people who work here are always in a jolly mood, always laughing and smiling amongst themselves and the customers. This Happy Lemon branch has a happy vibe and I like it."]
    #text=["Before I went on my trip to the Philippines I saw everyone all about this Macao Imperial Tea. I had to try it for myself since I love milk teas. I tried out the branch in SM Legaspi. I got the Cream Cheese Oreo Milk Tea. I loved every sip of it. It was very creamy, the sugar level was perfect and so did the ice and there was even oreo bits in the bottom of my cup."]
    #text=["Boba Tea has a great selection of teas and boba. The fruit teas aren't a sweet compared to other boba tea places I've been to which I appreciate. The staff is approachable and the space is clean.We'll most likely come back!"]
    #text=["No line at 3:30 on a Friday afternoon. We got the Thai tea and honey roasted milk teas. Both were delicious. The staff was very nice. There is a little parking lot, so parking was easy. I look forward to returning and trying their other teas and hot food items!"]
    #text=["Definitely would come back! SO MUCH SELECTION!! Lots of different options - Thai bubble tea, lychee strawberry tea, taro, matcha strawberry, etc.\r\n\r\nI went with the STRAWBERRY MATCHA & it was so so so delicious. The strawberry and matcha complimented each other and it still tasted so flavorful one did not overpower the other.\r\n\r\nI got mine with ALOE VERA + boba. The diff options for toppings aloe Vera / herbal jelly / chia seeds is really nice & different! I believe the teas are ~$6. Cheaper than mogee tea & to be honest I enjoyed this much better.\r\n\r\nThe outside seating area is cute! AND they have GAMES if you ask the bobarista (connect 4 / trouble / etc)\r\n\r\nDef a cute place to hang out and a delicious place to grab some good bubble tea'! They had some food options as well but didn't get any."]
    #from https://www.tripadvisor.com.ph/ShowUserReviews-g298452-d6966744-r251365618-HighTea_Milktea_Shop-Pasay_Metro_Manila_Luzon.html
    #text=["I've been a customer of hightea for a year now. Of all the milkteas I've tasted, I can say that I can't get enough of it. Why? First, hightea has this distinct taste, that when you drink their teas (especially the wintermelon milktea) you'll be back for more. As Katy Perry's song say, 'Comparisons are easily done once you have a taste of perfection.' As you try other milktea brands aside from Hightea, I am sure that you'll make comparisons, and you'll realize Hightea tastes good. Second, Hightea's service crew is the best!!! And when I say best, they're really the best!! It adds to the flavor of the milktea, I swear. They're the friendliest people I've met. If I were you, I will try this brand."]
    #text=["it taste bad"]
    #print("Review " + text[0])
    predict_sentiment(text)




    dbb = mysql.connector.connect(host='localhost',user='root',password='',database='Virginitea')
    cursor=dbb.cursor()
    print("Connecting to the database...")

    sentiment_query="SELECT * FROM `customer` WHERE sentiment = " + str(0)
    cursor.execute(sentiment_query)
    result=cursor.fetchall()
    print("Fetching database values...")

    feedback=[]
    user_ID=[]


    for x in result:
        user_ID.append(x[0])
        feedback.append([x[3]])

    x=0
    for elements in feedback:
        sentiment = predict_sentiment(feedback[x])
        query="UPDATE customer SET sentiment={} WHERE User_ID = {};".format(sentiment,user_ID[x])
        print("Querying...")
        cursor.execute(query)
        dbb.commit()
        x+=1

    print("Done you may now exit...")

    dbb.close()
    #cursor.execute("UPDATE customer SET sentiment WHERE
    #print(type(sentiment))


    # In[ ]:




