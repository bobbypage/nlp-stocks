from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
dateFormat = "yyyy-mm-dd"
input_file = "RedditNews.csv"
textblobTracker = {}
vaderTracker = {}

allData = []
with open(input_file) as fp:  
   line = fp.readline()
   cnt = 1
   print(line)
   currentLine = fp.readline()
   nextLine = " "
   while nextLine:
       nextLine = fp.readline()
       if len(nextLine) > len(dateFormat) and nextLine.startswith("20") and nextLine[len(dateFormat)] == ",":
            # this is the actual next line
            allData.append(currentLine)
            currentLine = nextLine
       else:
            # this needs to append to the current line 
            currentLine += nextLine

newsSentiment = open("newsSentiment.txt", "w")
newsSentiment.write("date, news, textBlobSentiment, vaderSentiment \n")

for line in allData:
    dateEndIndex = len(dateFormat)

    date = str(line[0:dateEndIndex])
    beginIndex = dateEndIndex + 1
    endIndex = len(line) - 1
    title = line[beginIndex:endIndex];
    title = title.replace('"', '')
    title = str(title)
    analysis = TextBlob(title)
    textblobPolarity = analysis.sentiment.polarity
    textBlobPolarityList = textblobTracker.get(date, [])
    textBlobPolarityList.append(textblobPolarity)
    textblobTracker[date] = textBlobPolarityList
    
    score = analyser.polarity_scores(title)
    vaderPolarity = score['compound']
    vaderpolarityList = vaderTracker.get(date, [])
    vaderpolarityList.append(vaderPolarity)
    vaderTracker[date] = vaderpolarityList
    seperator = ", "
    output = seperator.join([date, ' "' + title + '"', str(textblobPolarity), str(vaderPolarity)]) + "\n"
    newsSentiment.write(output)

averageSentimentPerDate = open("averageSentimentPerDate.txt", "w")
averageSentimentPerDate.write("date, textBlobSentimentAverage, vaderSentimentAverage")
for date in textblobTracker:
    textBlobPolarityList = textblobTracker[date]
    textBlobAve = sum(textBlobPolarityList) / len(textBlobPolarityList)
    vaderPolarityList = vaderTracker[date]
    vaderAve = sum(vaderPolarityList) / len(vaderPolarityList)
    
    seperator = ", "
    output = seperator.join([date, str(textBlobAve), str(vaderAve)]) + "\n"
    averageSentimentPerDate.write(output)


