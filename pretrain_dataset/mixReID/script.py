import os
import re
import shutil


def make_market_dir(root="./"):
    make_root = os.path.join(root, 'market1501/')
    train_dir = os.path.join(make_root, 'bounding_box_train')
    query_dir = os.path.join(make_root, 'query')
    test_dir = os.path.join(make_root, 'bounding_box_test')

    if os.path.exists(train_dir):
        os.makedirs(train_dir)

    if os.path.exists(query_dir):
        os.makedirs(query_dir)

    if os.path.exists(test_dir):
        os.makedirs(test_dir)


def extract_market1501(src_dir, dst_dir, **kwargs):
    imgNames = os.listdir(src_dir)
    pattern = re.compile(r'([-\d]+)_c(\d)')
    pid_container = set()
    for i, imgName in enumerate(imgNames):
        if '.jpg' not in imgName:
            continue
        print(imgName)
        # pid: 每个人的ID
        # _: 摄像头号
        pid, camid = map(int, pattern.search(imgName).groups())
        new_name = str(pid).zfill(6) + '_c' + str(camid) + '_mar' + str(i) + '.jpg'
        # 去掉没用的图像
        if pid == 0 or pid == 1:
            continue

        shutil.copy(os.path.join(src_dir, imgName), os.path.join(dst_dir, new_name))


def extract_cuhk03(src_dir, dst_dir, pid_num_cuhk):
    imgNames = os.listdir(src_dir)
    pattern = re.compile(r'([-\d]+)_c(\d)_([\d+])')
    pid_container = set()

    for i, imgName in enumerate(imgNames):
        if '.png' not in imgName and '.jpg' not in imgName:
            continue
        print(imgName)
        pid, camid, fname = map(int, pattern.search(imgName).groups())
        # print("cuhk03=====>{}".format(pid))
        # 需要加上market1501数据集的ID，因为ID从market1501后开始排
        pid += pid_num_cuhk
        # print("cuhk03_plus=====>{}".format(pid))
        dist_img_name = str(pid).zfill(6) + '_c' + str(camid) + '_cuhk' + str(i) + '.jpg'
        shutil.copy(os.path.join(src_dir, imgName), os.path.join(dst_dir, dist_img_name))


def extract_msmt17(dir_path, list_path, dst_dir, pid_num_msmt, m):
    with open(list_path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    if m == "train":
        dir_path = dir_path + "train"
    elif m == "query" or m == "test":
        dir_path = dir_path + "test"
    else:
        raise KeyError("Unsupported loss: {}".format(m))

    pid_container = set()
    for img_idx, img_infor in enumerate(lines):
        img_path, pid = img_infor.split(' ')
        # print("msmt=====>{}".format(pid))
        pid_ = int(pid) + pid_num_msmt
        # print("msmt_add=====>{}".format(pid_))
        camid = int(img_path.split('_')[2])
        img_path = os.path.join(dir_path, img_path)
        print(img_path)
        # name = img_path.split("/")[-1]
        newDir = os.path.join(dst_dir, str(pid_).zfill(6) + '_c' + str(camid) + '_msmt17' + str(img_idx) + '.jpg')
        shutil.copy(img_path, newDir)

    for idx, pid in enumerate(pid_container):
        assert idx == pid


def extract_dukemtmc(src_dir, dst_dir, pid_num_duke):
    imgNames = os.listdir(src_dir)
    pattern = re.compile(r'([-\d]+)_c(\d)_f([\d+])')
    pid_container = set()

    for i, imgName in enumerate(imgNames):
        if '.png' not in imgName and '.jpg' not in imgName:
            continue

        print(imgName)
        pid, camid, fname = map(int, pattern.search(imgName).groups())
        # 需要加上market1501数据集的ID，因为ID从market1501后开始排
        pid += pid_num_duke
        dist_img_name = str(pid).zfill(6) + '_c' + str(camid) + '_duke' + str(i) + '.jpg'
        shutil.copy(os.path.join(src_dir, imgName), os.path.join(dst_dir, dist_img_name))


if __name__ == "__main__":
    mix = r"/code/prsnet/pretrain_dataset/mixReID/train/"  # 混合、不带标签信息的数据保存路径，推荐绝对路径

    market1501 = {
        "train": r"/code/dataset/track/ReID/market1501/bounding_box_train",
        "test": r"/code/dataset/track/ReID/market1501/bounding_box_test",
        "query": r"/code/dataset/track/ReID/market1501/query"
    }

    cuhk03 = {
        "train": [r"/code/dataset/track/ReID/cuhk03-np/detected/bounding_box_train", 1501],
        "test": [r"/code/dataset/track/ReID/cuhk03-np/detected/bounding_box_test", 1501],
        "query": [r"/code/dataset/track/ReID/cuhk03-np/detected/query", 1501]
    }

    msmt17 = {
        "train": ["/code/dataset/track/ReID/msmt17/list_train.txt", 2968],
        "test": ["/code/dataset/track/ReID/msmt17/list_gallery.txt", 2968],
        "query": ["/code/dataset/track/ReID/msmt17/list_query.txt", 2968],
        "root": r"/code/dataset/track/ReID/msmt17/"
    }

    dukemtmc = {
        "train": [r"/code/dataset/track/ReID/dukemtmc/bounding_box_train", 7070],
        "test": [r"/code/dataset/track/ReID/dukemtmc/bounding_box_test", 7070],
        "query": [r"/code/dataset/track/ReID/dukemtmc/query", 7070]
    }

    for m in market1501:
        print("========> We're drawing Market1501 <=========")
        extract_market1501(src_dir=market1501[m], dst_dir=mix)

    for m in cuhk03:
        print("========> We're drawing CUHK03 <=========")
        extract_cuhk03(src_dir=cuhk03[m][0], dst_dir=mix, pid_num_cuhk=cuhk03[m][1])

    for m in msmt17:
        print("========> We're drawing MSMT17 <=========")
        if m == "root":
            continue
        extract_msmt17(dir_path=msmt17["root"], list_path=msmt17[m][0], dst_dir=mix, pid_num_msmt=msmt17[m][1], m=m)

    for m in dukemtmc:
        print("========> We're drawing DukeMTMC <=========")
        extract_dukemtmc(src_dir=dukemtmc[m][0], dst_dir=mix, pid_num_duke=dukemtmc[m][1])

    # make_market_dir(root='/code/dataset/track/ReID/mix_reid_datasets/')