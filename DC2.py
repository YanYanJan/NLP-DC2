import csv
from nltk.corpus import stopwords
import string
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

label = []
text = []

with open('spam.csv', encoding="ISO-8859-1") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            label .append(row["v1"])
            text.append(row["v2"])
            line_count += 1
        else:
            line_count += 1
            label .append(row["v1"])
            text.append(row["v2"])
            #if you uncommnet then it will print out the messages
            #print(row["v1"],row["v2"])

#print(f'Processed {line_count} lines.')


def clean_text(text):
    remove_punc = [char for char in text if char not in string.punctuation]
    remove_punc  = ''.join(remove_punc)
    clean_words = [word for word in remove_punc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

#vecotrize the inputs
vectorizer = TfidfVectorizer("english")
message = vectorizer.fit_transform(text)


# this does the spliting for the training and testing data
x_train, x_test, y_train, y_test = train_test_split(message, label, test_size = 0.2, random_state = 1)


# MultinomialNB Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
classifier = MultinomialNB()
classifier.fit(x_train, y_train)

nvb_pred = classifier.predict(x_test)
print(classification_report(y_test ,nvb_pred ))
print('Confusion Matrix: \n',confusion_matrix(y_test,nvb_pred))
print('Accuracy: ', accuracy_score(y_test,nvb_pred))


print()
print()

#Support Vector Machine
from sklearn import svm
svm = svm.SVC(C=1000)
svm.fit(x_train, y_train)

svm_pred = svm.predict(x_test)
print(classification_report(y_test, svm_pred ))
print("Confusion Matrix: \n'", confusion_matrix(y_test, svm_pred))
print('Accuracy: ', accuracy_score(y_test, svm_pred))

'''
def pred(msg):
    msg = vectorizer.transform([msg])
    prediction = svm.predict(msg)
    return prediction[0]

for i in range (11):
    print (text[i],pred(text[i]))

#logistic Regression

#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(x_train, y_train)
pred = Spam_model.predict(x_test)
print(accuracy_score(y_test,pred))
confusion_matrix_lr = pd.crosstab(y_test, pred)
print("Confusion Matrix: \n'",confusion_matrix_lr)
'''