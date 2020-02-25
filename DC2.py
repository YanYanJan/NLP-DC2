#import pandas as pd
import csv

# df = pd.read_csv(r'C:\Users\Alexis Hale\PycharmProjects\NLP-DC2\spam_notriplecommas.csv')   #read the csv file (put 'r' before the path string to address any special characters in the path, such as '\'). Don't forget to put the file name at the end of the path + ".csv"
# print(df)


with open('spam.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', lineterminator=',,,')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            print(row)
            line_count += 1
print(f'Processed {line_count} lines.')

# input_file = csv.DictReader(open("spam.csv"))

# for row in input_file:
#     print(row)
# else:
#     print("The file does not have any content")