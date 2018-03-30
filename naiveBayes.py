# This is the main file for my Naive Bayes classifier Assignment for
# my Data Mining Course.

import naiveBayesFunctions as nbf

trainPath = nbf.loadFile("train")
testPath = nbf.loadFile("test")

trainData, testData, flag = nbf.verifyData(trainPath, testPath)
if not trainData:
    if flag:
        print("The training and test data do not have the same attributes. Please select new files and re-run the program.")
        quit()
    else:
        print("There was a problem in parsing the file. Please re-run the program.")
        quit()

attributes = trainData[0]
print("Please choose an attribute by number:")
i = 1
for attribute in attributes:
    print("{0}. {1}".format(i, attribute))
    i = i + 1

target = input()
if int(target) > i-1:
    print("Target value not applicable. Please re-run the program and select a number between 1 and {}".format(i))
    quit()

dict, labels = nbf.identifyAttributes(trainData)
targetDict = nbf.train(trainData, int(target)-1, dict)

results = nbf.generateResults(testData, target, targetDict, labels)
print("The result is in the file Result.txt")
