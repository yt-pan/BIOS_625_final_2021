#!/usr/bin/env python3

import csv
import os

INPATH = 'C:\\Users\\panyt\\OneDrive\\Coursework\\BIOS_625\\final\\cleaned_var_csv\\'
OUTPATH = 'E:\\BIOS625final\\add_catdelay_csv\\'


def add_cat_delay():
    csv_files = ["2003.csv", "2004.csv", "2005.csv", "2006.csv", "2007.csv", "2008.csv"]
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
                    if row[0] == 'Year':
                        writer.writerow(row + ['depdelayC'])
                    else:
                        if int(row[9]) <= 15:
                            writer.writerow(row + ['0'])
                        else:
                            writer.writerow(row + ['1'])


def merge():
    csv_files = ["2003.csv", "2004.csv", "2005.csv", "2006.csv", "2007.csv", "2008.csv"]
    i = 0
    f_out ='D:\\' + '03-08.csv'
    allre = 0
    with open(f_out, 'w', newline='', encoding='utf-8') as opfile:
        writer = csv.writer(opfile)
        for csv_file in csv_files:
            i += 1
            f_in = OUTPATH + csv_file
            print("begin ", csv_file, i, "/", len(csv_files))
            with open(f_in, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] == 'Year':
                        if allre == 0:
                            writer.writerow(row)
                            allre = 1
                        else:
                            continue
                    else:
                        writer.writerow(row)


if __name__ == '__main__':
#    add_cat_delay()
    merge()
