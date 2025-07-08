"""
This is a template for generating sequence embeddings from the original public dataset,
using Assist0910 as an example.
"""

import csv
import src.configs as C
import json

def main():
    with open("assist0910_colname_to_index.json") as fh:
        COL_TO_INDEX = json.load(fh)

    exidarr = []
    correctarr = []
    firstflag = True

    with open(C.ASSIST0910_CLEAN_PATH, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if firstflag:
                curstudentId = row[COL_TO_INDEX['user_id']]
                firstflag = False
            if row[COL_TO_INDEX['user_id']] != curstudentId:
                with open(C.ASSIST0910_SEQ_PATH, "a+", newline='') as csvtowrite:
                    writer = csv.writer(csvtowrite)
                    writer.writerows([[len(exidarr)], exidarr, correctarr])
                exidarr.clear()
                correctarr.clear()
                curstudentId = row[COL_TO_INDEX['user_id']]
            exidarr.append(row[COL_TO_INDEX['problem_id']])
            correctarr.append(row[COL_TO_INDEX['correct']])
        with open(C.ASSIST0910_SEQ_PATH, "a+", newline='') as csvtowrite:
            writer = csv.writer(csvtowrite)
            writer.writerows([[len(exidarr)], exidarr, correctarr])


if __name__ == '__main__':
    main()
