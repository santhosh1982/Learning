list_12ns = []

for n in range(1, 34, 2):
    list_12n = []
    for i in range(0, 13):
        n1 = (12 * i + n) - (i*i)
        list_12n.append(n1)
    list_12ns.append(list_12n) 


for list12n in list_12ns:
    print(list12n)

