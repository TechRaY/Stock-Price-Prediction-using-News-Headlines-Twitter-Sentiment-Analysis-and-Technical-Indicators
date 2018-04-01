import csv
import numpy as np
import pandas as pd

df = pd.read_csv("tweets1.csv", index_col = 0)

mydict=set()
#print(df.iat[0,0])
length=len(df)

TCS = ["TCS","tcs","NatarajanChandrasekaran", "TATA","@tcs","@TCS"]
Infosys=["Infosys","Vishal Sikka","infosys","@infosys","@Infosys"]


for i in range(0,length):
	if not df.iat[i,0] in mydict:
		flag=0


		for j in Infosys:
			if j in df.iat[i,0]:
				strng=df.iat[i,0]+","+str(df.iat[i,1])+","+str(df.iat[i,2])+","+str(df.iat[i,3])
				#print(strng.encode("utf-8"))
				mydict.add(df.iat[i,0])
				flag=1
				break
	
		
		for j in TCS:
			if j in df.iat[i,0]:
				strng=df.iat[i,0]+","+str(df.iat[i,1])+","+str(df.iat[i,2])+","+str(df.iat[i,3])
				#print(strng.encode("utf-8"))
				mydict.add(df.iat[i,0])
				flag=1
				break
	
		if flag==0:		
			strng=df.iat[i,0]+","+str(df.iat[i,1])+","+str(df.iat[i,2])+","+str(df.iat[i,3])
			mydict.add(df.iat[i,0])
			print(strng.encode("utf-8"))
			
