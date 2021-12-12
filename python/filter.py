#!/usr/bin/env python3

import csv
import os

INPATH = 'C:\\Users\\panyt\\Downloads\\Compressed\\DataExpo2009\\raw_csv\\'
OUTPATH = 'C:\\Users\\panyt\\Downloads\\Compressed\\DataExpo2009\\cleaned_var_csv\\'


# 1 2 3 4 6 8 9 10 15 16 17 18 19 22==0 24==0
def main_sep():
    csv_files = os.listdir(INPATH)
    i = 0
    for csv_file in csv_files:
        i += 1
        f_in = INPATH + csv_file
        f_out = OUTPATH + csv_file
        print("begin ", csv_file, i, "/", len(csv_files))
        with open(f_in, 'r', newline='', encoding='utf-8') as csvfile:
            with open(f_out, 'w', newline='', encoding='utf-8') as opfile:
                reader = csv.reader(csvfile)
                writer = csv.writer(opfile)
                for row in reader:
                    newrow = row[0:4] + [row[5]] + row[7:10] + row[14:19]
                    if row[0] == 'Year':
                        writer.writerow(newrow)
                    else:
                        if row[21] == '0' and row[23] == '0':
                            writer.writerow(newrow)


def main_merge():
    csv_files = os.listdir(INPATH)
    i = 0
    f_out = OUTPATH + 'Full.csv'
    allre = 0
    with open(f_out, 'w', newline='', encoding='utf-8') as opfile:
        writer = csv.writer(opfile)
        for csv_file in csv_files:
            i += 1
            f_in = INPATH + csv_file
            print("begin ", csv_file, i, "/", len(csv_files))
            with open(f_in, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    newrow = row[0:4] + [row[5]] + row[7:10] + row[14:19]
                    if row[0] == 'Year':
                        if allre == 0:
                            writer.writerow(newrow)
                            allre = 1
                        else:
                            continue
                    else:
                        if row[21] == '0' and row[23] == '0':
                            writer.writerow(newrow)


if __name__ == '__main__':
    main_merge()
    main_sep()
