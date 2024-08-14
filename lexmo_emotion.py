# from LeXmo import LeXmo
# text = """From the beginning, she had sat looking at him fixedly.\n  As he now leaned back in his chair, and bent his deep-set eyes upon her in his turn,\n  perhaps he might have seen one wavering moment in her, \n  when she was impelled to throw herself upon his breast,\n  and give him the pent-up confidences of her heart.\n  But, to see it, \n  he must have overleaped at a bound the artificial barriers he had for many years been erecting, \n  between himself and all those subtle essences of humanity which will elude the utmost cunning of algebra\n  until the last trumpet ever to be sounded shall blow even algebra to wreck.\n  The barriers were too many and too high for such a leap. With his unbending,\n  utilitarian, matter-of-fact face, he hardened her again;\n  and the moment shot away into the plumbless depths of the past,\n  to mingle with all the lost opportunities that are drowned there"""
# emotion = LeXmo.LeXmo(text)
# emotion.pop(text,None)
# print(emotion)


import pandas as pd
import numpy as np
import seaborn as sns
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

df = pd.read_csv(r'C:\Users\swath\OneDrive\Desktop\Swathika\MeisterGen\EmotionDetection\emotion_dataset_2.csv')
#print(df.head())
sns.countplot(x='Emotion',data=df)

df['clean_text'] = df['Text'].apply(nfx.remove_userhandles)
df['clean_text'] = df['clean_text'].apply(nfx.remove_stopwords)
df['clean_text'] = df['clean_text'].apply(nfx.remove_special_characters)

#print(df.head())

X =df['clean_text']
y = df['Emotion']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])
pipe_lr.fit(X_train,y_train)
print("Emotions:",pipe_lr.classes_)
print("Score:",pipe_lr.score(X_test,y_test))

pred = input("Enter your text:")
print("\nPredicted emotion:",pipe_lr.predict([pred]))