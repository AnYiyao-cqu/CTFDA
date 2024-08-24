
import os
_dir = r"E:\CTFDA\data\IMS"


def get_file(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    if len(file_list) != 1:  # 若文件中存在不止一个文件，则存在歧义
        print('There are {} files in [{}]'.format(len(file_list), root_path))
        exit()
    return file_list[0]


# NC
NC = get_file(os.path.join(_dir, r'NC'))

# IF
IF = get_file(os.path.join(_dir, r'IF'))


# OF
OF = get_file(os.path.join(_dir, r'OF'))

# RoF
ROF = get_file(os.path.join(_dir, r'ROF'))

# Tasks  ROF ,
T_IMS = [NC, OF, IF]



if __name__ == "__main__":
    print(T_IMS)
    pass
