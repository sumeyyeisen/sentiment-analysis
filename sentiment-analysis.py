'''
Veri Seti
Veri seti çok büyük olduğu için GitHub'a yüklenememiştir.  
İndirmek için: [Dataset’i Google Drive’dan indir](https://drive.google.com/file/d/1j7h7dDilWn9ObrQYPgBtBdumrx836-xH/view?usp=drive_link)
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')  

data = pd.read_csv('data/training.1600000.processed.noemoticon.csv', header=None, encoding='latin1')
data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text'] 
data.drop(['query'], axis=1, inplace=True) 
data.dropna(subset=['text', 'sentiment'], inplace=True)

data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 4 else 0)  

words = set(stopwords.words('english')) 
data['text'] = data['text'].str.lower() 
data['text'] = data['text'].str.replace(r'http\S+|www\S+', '', regex=True) 
data['text'] = data['text'].str.replace(r'@\w+', '', regex=True) 
data['text'] = data['text'].str.replace(r'#\w+', '', regex=True)
data['text'] = data['text'].str.replace(r'[^a-z\s]', '', regex=True)  
data['text'] = data['text'].str.strip() 
data['text'] = data['text'].str.replace(r'\s+', ' ', regex=True) 

data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in words])) 

print(data.head()) 

vector = TfidfVectorizer(max_features=5000) 
X = vector.fit_transform(data['text'])
y = data['sentiment'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)


y_predict = model.predict(X_test) 
y_predict_prob = model.predict_proba(X_test)[:, 1] 


accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

print("Doğruluk:", accuracy)
print("Kesinlik:", precision)
print("Duyarlılık:", recall)
print("F1-Skoru:", f1)


matrix = confusion_matrix(y_test, y_predict)
color= sns.light_palette("pink", as_cmap=True)
sns.heatmap(matrix, annot=True, fmt='d', cmap=color, xticklabels=['Pozitif', 'Negatif'], yticklabels=['Pozitif', 'Negatif'])
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()


fpr, tpr, score_limits = roc_curve(y_test, y_predict_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', lw=2, label='ROC Eğrisi (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()

print(f"AUC Skoru: {roc_auc}")


precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_predict_prob)
average_precision = average_precision_score(y_test, y_predict_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PRC Eğrisi (AP = {average_precision:.2f})')
plt.xlabel('Recall (Duyarlılık)')
plt.ylabel('Precision (Kesinlik)')
plt.title('Precision-Recall Eğrisi')
plt.legend(loc="lower left")
plt.show()
