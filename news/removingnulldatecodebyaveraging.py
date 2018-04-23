import csv
import pandas as pd
import math

infy=pd.read_csv("input/alldatainfy.csv")

date=infy.Date
data=infy.Open	

#print(len(data))
contvalue=[]
prev=0

for i in range(len(data)):
	
	if not math.isnan(data[i]):
		prev=data[i]
		print("rowno "+str(i))
		contvalue.append(data[i])
	
	else:
		
		count=0
		
		print("rowno hi "+str(i))
		while(math.isnan(data[i])):
			i=i+1
			print("rowno hi"+str(i))
			count+=1
			
		current=data[i]
		adder=abs(current-prev)
		count+=1
		adder=adder/count
		
		while(count>0):
			prev=prev+adder
			contvalue.append(prev)
			count-=1
		
		contvalue.append(current)	
		prev=current
		
		
#print(contvalue)