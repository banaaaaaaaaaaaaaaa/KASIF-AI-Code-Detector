myFile = open('expenses.txt', mode = "w")
while True:
    x = str(input())
    if x != "-1":
        
        myFile.write(x)
        myFile.write("\n")
    else:
        break

myFile = open("expenses.txt", mode = "r")
a = myFile.read()
myList = []
myList.append(int(myFile.read()))
avg = sum(myList)/len(myList)
print('Avg:',avg)
