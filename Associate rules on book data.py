#Some modification is done to book data in excel and saved as book_new...
#and i have send the modified excel sheet along with association rules answer folder..

import numpy  as np
import pandas as pd

df=pd.read_csv("book_new.csv")
df.head()
df.info()
df.shape
df.values[0]

Excelr=[]

#To use Apriori Algorithm the data should be in list..

for i in range(0, 2000):
  Excelr.append([str(df.values[i,j]) for j in range(0, 11)])
Excelr
type(Excelr)
len(Excelr)

########

from apyori import apriori
rules = apriori(transactions = Excelr, min_support = 0.03, min_confidence = 0.5, min_lift = 2, min_length = 2,max_length=3)

rules

results=list(rules)

def inspect(results):
    book1         = [tuple(result[2][0][0])[0] for result in results]
    book2       = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(book1,book2, supports, confidences, lifts))

rules = pd.DataFrame(inspect(results), columns = ['book1', 'book2', 'Support', 'Confidence', 'Lift'])

rules

#observation:
    
#for min_support = 0.003, min_confidence = 0.2, min_lift = 2, min_length = 2,max_length=2 then 13 Rules are generated..
#for min_support = 0.03, min_confidence = 0.2, min_lift = 2, min_length = 2,max_length=2 then 8 Rules are generated...
#for min_support = 0.03, min_confidence = 1, min_lift = 2, min_length = 2,max_length=2 then 3 Rules are generated...
#for min_support = 0.03, min_confidence = 0.5, min_lift = 2, min_length = 2,max_length=2 then 7 Rules are generated...
#changing min length
#for min_support = 0.03, min_confidence = 1, min_lift = 2, min_length = 3,max_length=4 then 40 Rules are generated....
