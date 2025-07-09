import csv
import random
import src.configs as C

def load_data(fileName):
    rows = []
    max_skill_num = 0
    # max_num_problems = 0
    max_num_problems = 500
    with open(fileName, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
    print("filename: " + str(fileName) + "the number of rows is " + str(len(rows)))

    index = 0
    tuple_rows = []
    while(index < len(rows)-1):
        problems_num = int(rows[index][0])
        tmp_max_skill = max(map(int, rows[index+1]))
        if(tmp_max_skill > max_skill_num):
            max_skill_num = tmp_max_skill
        if(problems_num <= 2): #discard
            index += 3
        else:
            if problems_num > max_num_problems:
                # max_num_problems = problems_num
                tup = (rows[index], rows[index + 1][:max_num_problems], rows[index + 2][:max_num_problems])
            else:
                tup = (rows[index], rows[index+1], rows[index+2])
            # tup:[num_qs, q_seq, correctness]
            tuple_rows.append(tup)
            index += 3

    # shuffle the tuple
    # random.seed(420)
    # random.shuffle(tuple_rows)
    # return tuple_rows, max_num_problems, max_skill_num+1
    return tuple_rows, max_skill_num + 1