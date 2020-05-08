import re
import os
import csv
import sys

dirs = ["img/original", "img/modified"]

try:
    with open('img/english_subset.csv', 'a', newline='') as english_csv, \
            open('img/mandarin_subset.csv', 'a', newline='') as mandarin_csv, \
            open('img/korean_subset.csv', 'a', newline='') as korean_csv:

        english_writer = csv.writer(english_csv)
        english_writer.writerow(["file_name", "native_language"])

        mandarin_writer = csv.writer(mandarin_csv)
        mandarin_writer.writerow(["file_name", "native_language"])

        korean_writer = csv.writer(korean_csv)
        korean_writer.writerow(["file_name", "native_language"])

        writer_dict = {"english": english_writer, "mandarin": mandarin_writer, "korean": korean_writer}

        for dir in dirs:
            path = os.getcwd()+"/"+dir
            for filename in os.listdir(path):
                r = re.compile("([e-m])\D+")
                m = r.search(filename)
                writer_dict[m.group(0)].writerow([filename, m.group(0)])
except IOError as e:
    errno, strerror = e.args
    print("I/O error({0}): {1}".format(errno, strerror))
except Exception as other_exception:
    errno, strerror = other_exception.args
    print("Unexpected error:".format(errno, strerror), sys.exc_info()[0])
    raise
