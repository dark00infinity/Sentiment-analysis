import main_sentiment
import matplotlib.pyplot as plt
import pandas as pd
import sys
new_words = {'below':-2,'above':2}
analyser=main_sentiment.Sentiments()
hero=int(input("enter number of sentences"))
analyser.lexicon.update(new_words)
dataset=pd.read_csv("Reviews.tsv", delimiter="\t")
review=dataset.iloc[:,0]
count,count1,count2,count3,count4=0,0,0,0,0
for i in range(hero):
    sentiment=analyser.review_scores(review[i])
    y=[sentiment["pos"],sentiment["neu"],sentiment["neg"],sentiment["compound"]]
    tlabel=["positive","neutral","negative","compound"]
    left=[1,2,3,4]
    plt.bar(left,y,tick_label=tlabel,width=0.5,color=["green","yellow","red","blue"])
    plt.show()
    k=y.pop()
    y.sort()
    if sentiment["compound"]>=0.33:
            print("Positive")
            hi=open("positive.txt","a")
            hi.write(review[i]+"\n")
            hi.close()
            count=count+1
    elif sentiment["compound"]<(-0.33):
            print("Negative")
            hi=open("negative.txt","a")
            hi.write(review[i]+"\n")
            hi.close()
            count1=count1+1
    elif sentiment["compound"]==0:
            print("Neutral")
            hi=open("Neutral.txt","a")
            hi.write(review[i]+"\n")
            hi.close()
            count2=count2+1
    elif sentiment["compound"]>0 and sentiment["compound"]<0.33:
            print("Neutral positive")
            hi=open("neutralpositive.txt","a")
            hi.write(review[i]+"\n")
            hi.close()
            count3=count3+1
    else:
        print("Neutral negative")
        hi=open("neutralnegative.txt","a")
        hi.write(review[i]+"\n")
        hi.close()
        count4=count4+1
if count>count1 or count>count2 or count>count3 or count>count4:
    per1=(count*100)/hero
    print(per1,"overall positive")
elif count1>count or count1>count2 or count1>count3 or count1>count4:
    per2=(count1*100)/hero
    print(per2,"overall negative")
elif count2>count or count2>count1 or count2>count4 or count2>count3:
    per3=(count2*100)/hero
    print(per3,"overall neutral")
elif count3>count or count3>count1 or count3>count2 or count3>count4:
    per4=(count3*100)/hero
    print(per4,"overall neutral positive")
else:
    per5=(count4*100)/hero
    print(per5,"overall neutral negative")
per1=(count*100)/hero
per2=(count1*100)/hero
per3=(count2*100)/hero
per4=(count3*100)/hero
per5=(count4*100)/hero
    
    
y=[per1,per2,per3,per4,per5]
tlabel=["positive","negative","neutral","neupositive","neunegative"]
left=[1,2,3,4,5]
plt.bar(left,y,tick_label=tlabel,width=0.5,color=["green","yellow","red","blue","orange"])
plt.show()
k=y.pop()
y.sort()