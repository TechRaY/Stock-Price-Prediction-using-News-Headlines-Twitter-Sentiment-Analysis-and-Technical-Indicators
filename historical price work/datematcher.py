#required files are in input folder
import csv

file2reader = csv.reader(open("/input/infyalldate.csv"), delimiter=",")
header2 = file2reader.next() #header

mydict={}
raj=0

for DateAll,Open1,High1,Low1,Close1,Adj_Close1,Volume1 in file2reader:
	file1reader = csv.reader(open("infy/INFY.csv"), delimiter=",")	
	header1 = file1reader.next() #header
	for Date,Open,High,Low,Close,Adj_Close,Volume in file1reader:
		if Date==DateAll: 
			mydict[Date]=Open
			print(Date+","+Open+","+High+","+Low+","+Close+","+Adj_Close+","+Volume)
			break
			
		elif mydict.has_key(Date)==False:
			mydict[DateAll]="null"
			print(DateAll+",null"+",null"+",null"+",null"+",null"+",null")
			break
		
		else :
			continue


#run python datamatcher.py >filenametostore