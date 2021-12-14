#!/usr/bin/env python3

import csv
import random

INPATH = 'C:\\Users\\panyt\\OneDrive\\Coursework\\BIOS_625\\final\\cleaned_var_csv\\'
OUTPATH = 'E:\\BIOS625final\\add_catdelay_csv\\'

def sel_testset():
    f_in = 'D:\\' + '03-08.csv'
    f_out = 'D:\\' + 'debug.csv'
    with open(f_out, 'w', newline='', encoding='utf-8') as opfile:
        writer = csv.writer(opfile)
        with open(f_in, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',' )
            for row in reader:
                k = random.uniform(0,1)
                if row[0] == 'Year':
                    writer.writerow(row)
                else:
                    if k<=0.001:
                        writer.writerow(row)           


if __name__ == '__main__':
    sel_testset()
