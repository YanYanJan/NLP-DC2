import pandas as pd
import csv
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# df = pd.read_csv(r'C:\Users\Alexis Hale\PycharmProjects\NLP-DC2\spam_notriplecommas.csv')   #read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
# print(df)

#store the spam or ham label and actual content
from sklearn.model_selection import train_test_split

v1 = []
v2 = []

with open('spam.csv', encoding="ISO-8859-1") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            v1.append(row["v1"])
            v2.append(row["v2"])
            line_count += 1
        else:
            line_count += 1
            v1.append(row["v1"])
            v2.append(row["v2"])
            #if you uncommnet then it will print out the messages
            #print(row["v1"],row["v2"])

print(f'Processed {line_count} lines.')

#print(v2[0:5])

def clean_text(text):
    remove_punc = [char for char in text if char not in string.punctuation]
    remove_punc  = ''.join(remove_punc)
    clean_words = [word for word in remove_punc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

#print(clean_text(v2[2]))
# to use a logistic regression, we first convert all the data in to
#vectorizer = TfidfVectorizer("english")
#message = vectorizer.fit_transform(v2)
#print(message)

# this does the spliting for the training and testing data
#m_train, m_test, sns_train, sns_test = train_test_split(message,
#                                                        v1, test_size=0.3, random_state=20)
x_train, x_test, y_train, y_test = train_test_split(v2, v1, test_size = 0.2, random_state = 1)


vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)


#logistic Regression
'''
Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(x_train, y_train)
pred = Spam_model.predict(x_test)
print(accuracy_score(y_test,pred))
confusion_matrix_lr = pd.crosstab(y_test, pred)
print(confusion_matrix_lr)
'''

#Support Vector Machine
svm = svm.SVC(C=1000)
svm.fit(x_train, y_train)

X_test = vectorizer.transform(x_test)
y_pred = svm.predict(X_test)
print(confusion_matrix(y_test, y_pred))



# input_file = csv.DictReader(open("spam.csv"))

# for row in input_file:
#     print(row)
# else:
#     print("The file does not have any content")

