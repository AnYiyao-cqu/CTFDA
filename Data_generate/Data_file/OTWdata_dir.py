
import os
_dir = r"E:\CTFDA\data\Ottawa"


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


# Tasks
T_OTW = [NC, IF, OF]



if __name__ == "__main__":
    print(T_OTW)
    pass
