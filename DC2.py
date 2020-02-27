#import pandas as pd
import csv
import nltk
from nltk.corpus import stopwords
import string


# df = pd.read_csv(r'C:\Users\Alexis Hale\PycharmProjects\NLP-DC2\spam_notriplecommas.csv')   #read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
# print(df)

#store the spam or ham label and actual content
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
    remove_punc  = ''.join(remove_punc )
    clean_words = [word for word in remove_punc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

print(clean_text(v2[2]))




# input_file = csv.DictReader(open("spam.csv"))

# for row in input_file:
#     print(row)
# else:
#     print("The file does not have any content")