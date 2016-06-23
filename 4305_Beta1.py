import os
import io
import re
import nltk
from collections import Counter
from bs4 import BeautifulSoup
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# create counter object (dictionary like hash..)
counter_train = Counter()
counter_test = Counter()

# create porter stemmer object
porter = nltk.PorterStemmer()

# set up empty lists
training_set = []
test_set = []

# number of documents
testDoc=0
trainDoc=0
testcourse=0
testfaculty=0
teststudent=0
traincourse=0
trainfaculty=0
trainstudent=0

# list of stop words
stop_words = set(stopwords.words('english'))

print("Begin script...")

# root directory we traverse
directory = 'datasets/'

# traverse all sub-dirs/docs
for root, dirs, files in os.walk(directory):
    for name in files:
        filename = os.path.join(root, name)

        # open file stream
        document = io.open(filename, encoding='latin-1')

        # parse document using lxml
        html = BeautifulSoup(document, 'lxml')

        # close file stream
        document.close

        # get raw text of document
        if html.find('htmlplus'):  # to deal with the weird html tags
            raw = html.htmlplus.get_text()
        else:
            html.html.p.extract()  # for rest of the documents
            raw = html.html.get_text()

        # lower capitalization
        raw = raw.lower()

        # tokenize into words
        tokens = word_tokenize(raw)

        # keep words greater than 3 characters
        tokens = [t for t in tokens if len(t) >= 3]

        # remove non-alphanumeric words
        tokens = [t for t in tokens if re.search('[^a-zA-Z-]', t) == None]

        # remove stop words
        filtered_words = filter(lambda token: token not in stop_words, tokens)

        # stem filtered word and add it to the proper list
        if "datasets/train" in root:
            trainDoc=trainDoc+1
            training_set = [porter.stem(word) for word in filtered_words]
            counter_train.update(training_set)
            if "course" in root:
                testcourse+=1
            if "faculty" in root:
                testfaculty+=1
            if "student" in root:
                teststudent+=1
            
        if "datasets/test" in root:
            testDoc=testDoc+1  #for TD-IDF
            test_set = [porter.stem(word) for word in filtered_words]
            counter_test.update(test_set)
            if "course" in root:
                traincourse+=1
            if "faculty" in root:
                trainfaculty+=1
            if "student" in root:
                trainstudent+=1
            
        
       
            
            
            

# frequency of a word occurring in the data set
fdist_train = FreqDist(counter_train)
fdist_test = FreqDist(counter_test)
fdist_both = FreqDist(counter_test + counter_train)

# output
print("")
print(str(len(counter_train)) +
      " unique words are identified from the training set")
print(str(len(counter_test)) +
      " unique words are identified from the test set")
print("")
print("Frequency of 200 most common words in training data:")
print(fdist_train.most_common(200))
print("")
print("Frequency of 200 most common words in testing data:")
print(fdist_test.most_common(200))
print("")
print("Frequency of 200 most common words in both set:")
print(fdist_both.most_common(200))
print("")
print(str(len(counter_test | counter_train)) +
      " unique words are identified from both set/ all set")
print(str(len(counter_test & counter_train)) +
      " identical words are appeared on both set/ all set (intersection)")
test_notTrain = len(set(counter_test) - set(counter_train))
train_notTest = len(set(counter_train) - set(counter_test))
print(str(test_notTrain) + " words are appeared in train ,but not test")
print(str(train_notTest) + " words are appeared in test ,but not train")
print(str(test_notTrain + train_notTest) + " are not appeared in both set.")


print(str(testDoc) +" is in the test folder")
print(str(trainDoc)+" is in the train folder")
print(str(testcourse) +" testcourse")
print(str(testfaculty) +" testfaculty")
print(str(teststudent) +" teststudent")
print(str(traincourse) +" traincourse")
print(str(trainfaculty) +" trainfaculty")
print(str(trainstudent) +" trainstudent")

print("\nEnd.\n")

