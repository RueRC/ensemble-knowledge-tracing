"""
This is a template for train-test split from the processed dataset,
using Assist0910 as an example.
"""

import csv

def main():
    # get total rows
    totalrow = 0
    with open("assist0910_seq.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            totalrow += 1
        csvfile.close()

    # write train file
    cur = 0
    with open("assist0910_seq.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            cur += 1
            if cur <= 0.8 * totalrow:
                with open("assist0910_seq_train.csv", "a+", newline='') as csvtrain:
                    writer = csv.writer(csvtrain)
                    writer.writerow(row)
            else:
                with open("assist0910_seq_test.csv", "a+", newline='') as csvtest:
                    writer = csv.writer(csvtest)
                    writer.writerow(row)


if __name__ == '__main__':
    main()
