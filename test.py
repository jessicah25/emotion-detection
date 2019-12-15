import csv

new = open("train_sent_emo_norm.csv")
reader = csv.reader(new, delimiter=',')

old = open("meld_xvectors.csv")
reader2 = csv.reader(old, delimiter=',')

du = []
line_count = 0
count = 0
for row in reader:
    if line_count == 0:
        line_count += 1
    else:
        du.append(str(row[5]) + "_" + str(row[6]))
        count += 1

count1 = 0
for row in reader2:
    if line_count == 1:
        line_count += 1
    else:
        if row[3] == "train":
            du.remove(str(row[0]) + "_" + str(row[1]))
            count1 += 1

print(du)
