import os
import shutil
import csv
import string

folder_path = "C:/Users/sirch/PycharmProjects/TFTutorials/venv/dataset/validation"
csv_path = "C:/Users/sirch/PycharmProjects/TFTutorials/venv/dataset/validation/GT-final_test.csv"

with open(csv_path) as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        tmp_filename = row["Filename"].split(".")[0]
        jpg_filename = tmp_filename+".jpg"
        destination = folder_path+"/{}".format(row["ClassId"])
        source = folder_path+"/{}".format(jpg_filename)
        done = shutil.move(source, destination)
        print(done)