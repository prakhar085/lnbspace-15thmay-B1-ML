###Question1
# l = list(map(int, input().split()))(for taking custom inputs)
print("Answer 1: ", end=" ")
l = [386, 462, 47, 418, 907, 344, 236, 375, 823, 566, 597, 978, 328, 615,
     953, 345, 399, 162, 758, 219, 918, 237, 412, 566, 826, 248, 866, 950, 626, 949,
     687, 217, 815, 67, 104, 58, 512, 24, 892, 894, 767, 553, 81, 379, 843, 831, 445,
     742, 717, 958, 743, 527]
for i in l:
    if i == 237:
        break
    elif i % 2 == 0:  # Checking if given number is even
        print(i, end=" ")
print(*[])

###Question 2
print("Answer 2: ", end=" ")
s1 = {"White", "Black", "Red"}
s2 = {"Red", "Green"}
res = set()
for j in s1:
    if j not in s2:
        res.add(j)  # Checking if given item is present in both s1 and s2
print(*list(res))
# print("\n")


###Question 3
print("Answer 3: ", end=" ")


def pangram(s):
    s = s.lower()
    al = "abcdefghijklmnopqrstuvwxyz"
    for j in al:
        if j not in s:
            return False
    return True


s = "argfefegbefggtgcgsghyjdttyjtyjeyhyfrryherygthrthhijklmnopqrstuxwxyz"  # We can take custom input as well
print(pangram(s))

###Question 4
print("Answer 4: ", end=" ")

n = 5  # we can take custom input as well (int(input())
print(*[n + int(str(n) * 2) + int(str(n) * 3)])

###Question 5
print("Answer 5: ", end=" ")
p = "23 24 45#345 34 12"
l = list(map(str, p.split()))
l1 = [int(l[0]), int(l[1]), int(l[2][:l[2].index("#")])]
l2 = [int(l[2][l[2].index("#") + 1:]), int(l[3]), int(l[4])]
print("l1: ", *l1)
print(" " * 11 + "l2: ", *l2)

###Question 6
print("Answer 6: ", end=" ")
p = "without,hello,bag,world"
l = sorted(list(map(str, p.split(","))))
print(*l)

###Question 7
print("Answer 7: ", end=" ")

l = {'Student': ['Rahul', 'Kishore', 'Vidhya', 'Raakhi'],
     'Marks': [57, 87, 67, 56]}
n = len(l["Marks"])
maxi = max(l["Marks"])
student = []
for j in range(n):
    if l["Marks"][j] == maxi:
        student.append(l["Student"][j])
print(*student)

###Question 8
print("Answer 8: ", end=" ")
# l = input()
l = "Hello world!123"
let = 0
dig = 0
l = l.lower()
letters = "abcdefghijklmnopqrstuvwxyz"
digits = "0123456789"
for j in l:
    if j in letters:
        let += 1
    elif j in digits:
        dig += 1
print("LETTERS {}".format(let))
print(" " * 11 + "DIGITS {}".format(dig))

###Question 9
print("Answer 9: ", end=" ")


def specific(df, subj):
    n = len(df["Subject"])
    df2 = {"Name": [],
           "Subject": [],
           "Ratings": []}
    for i in range(n):
        if df["Subject"][i] == subj:
            df2["Name"].append(df["Name"][i])
            df2["Subject"].append(df["Subject"][i])
            df2["Ratings"].append(df["Ratings"][i])
    return df2


df = {'Name': ['Akash', 'Soniya', 'Vishakha', 'Akshay', 'Rahul', 'Vikas'],
      'Subject': ['Python', 'Java', 'Python', 'C', 'Python', 'Java'],
      'Ratings': [8.4, 7.8, 8, 9, 8.2, 5.6]}
subj = "Python"
df2 = specific(df, subj)
print(df2)

###Question 10:
print("Answer 10: ", end=" ")


class Generator:
    def __init__(self):
        self.l = []

    def solve(self, n):
        for i in range(1, n):
            if i % 7 == 0:
                self.l.append(i)
        print(*self.l)


s = Generator()
s.solve(100)



###Question 11
print("Answer 11: ", end=" ")
import math

i, j = 0, 0

# This snippet of code keeps on reading and solving until it gets a blank (Value error or EOFerror)
# for custom inputs

'''
while True:
    try:
        x, y = map(str, input().split())
        if x == "UP":
            j += int(y)
        elif x == "DOWN":
            j -= int(y)
        elif x == "LEFT":
            i -= int(y)
        else:
            i += int(y)
    except ValueError or EOFError:
        break
'''
##For the sake of this question when we have only 4 inputs we can simply run a for loop.
inpt = [["UP", "5"], ["DOWN", "3"], ["LEFT", "3"], ["RIGHT", "2"]]
for k in range(4):
    x, y = inpt[k][0], inpt[k][1]
    if x == "UP":
        j += int(y)
    if x == "DOWN":
        j -= int(y)
    if x == "LEFT":
        i -= int(y)
    if x == "RIGHT":
        i += int(y)
print(int(math.sqrt(i * i + j * j)))
