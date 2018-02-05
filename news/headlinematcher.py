import csv

filereader2 = csv.reader(open("input/infyalldate.csv"), delimiter=",")
header2 = filereader2.next() #header

for Date in filereader2:
	
	data=[Date[0]]
	filereader = csv.reader(open("input/finalsortedmoneycontrol.csv"), delimiter=",")
	header = filereader.next() #header
	flag=0
	
	for time,date,Source,News_Headlines in filereader:
		
		if Date[0]==date:
			data.append(News_Headlines)
			flag=1
			continue
			
		if Date[0]!=date:
			if flag==1:
				break
				
	print( ", ".join( repr(e) for e in data ) )
