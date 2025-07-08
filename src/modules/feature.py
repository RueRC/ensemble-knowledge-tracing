import csv
import os
import time
import numpy as np
import src.configs as C


def load_feature(file_name, num_of_data):
    '''
    :param file_name: file's name(assistment2017.csv)
    :param num_of_data: the row's number that you need in your datasets
    :return: list
    '''
    collection = []
    datasets = []
    last_studentid = 0
    num_current = 0
    store_id_flag = 1
    start_time = time.time()
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        print("Successfully get the csv file")
        for row in reader:
            if store_id_flag == 1:
                last_studentid = int(row[0])
                store_id_flag = 0
            if int(row[0]) == last_studentid:
                if num_current < num_of_data:
                    datasets.append(row)
                    num_current += 1
            else:
                collection.append(datasets)
                datasets = []
                last_studentid = int(row[0])
                num_current = 0
                datasets.append(row)
                num_current += 1
    collection.append(datasets)
    print("shape is ", np.shape(collection))
    end_time = time.time()
    print("Successfully process all the datas in ", end_time - start_time, " s.")
    return collection
