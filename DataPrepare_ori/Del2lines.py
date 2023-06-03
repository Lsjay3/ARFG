# Jay的开发时间：2022/11/5  10:41
import os

base_directory = 'E:\\笨比J\\RFID\\Impinj R420\\Data\\原始数据\\d'

for dir_path, dir_name_list, file_name_list in os.walk(base_directory):
    for file_name in file_name_list:

        # If this is not a CSV file
        if not file_name.endswith('.csv'):
            # Skip it
            continue
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'r') as ifile:
            line_list = ifile.readlines()
        with open(file_path, 'w') as ofile:
            ofile.writelines(line_list[2:])