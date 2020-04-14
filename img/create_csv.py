import re
import os
import csv

def get_label(argument):
    switcher = {
        "english": 0,
        "mandarin": 1,
        "korean": 2
    }
    return switcher.get(argument, 0)

with open('subset.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["file_name", "native_language"])
    for filename in os.listdir(os.getcwd()+"/original"):
        r = re.compile("([e-m])\D+")
        m = r.search(filename)
        writer.writerow([filename, m.group(0)])
