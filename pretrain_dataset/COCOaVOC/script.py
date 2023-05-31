import os
import xml.etree.ElementTree as ET

import cv2


def cut_image_save(img_path, save_dir, data_type, num, xmin, ymin, w, h):
    print(img_path)

    img = cv2.imread(img_path)

    cut_img = img[ymin:ymin + h, xmin:xmin + w, :]
    cv2.imwrite(os.path.join(save_dir, data_type + "_" + str(num) + ".jpg"), cut_img)


if __name__ == "__main__":
    imgs_path = "../../JPEGImages/"  # 图片源路径
    xmls_path = "../../Annotations/"  # 图片对应的标注文件
    data_type = "gy"  # 数据集类型，coco or voc

    save_dir = "../COCOaVOC/train/"

    num = 0

    for f in os.listdir(xmls_path):
        if f.split(".")[1] != "xml":
            continue
        xml_file = os.path.join(xmls_path, f)
        file_name = f.split(".")[0]
        img_file = os.path.join(imgs_path, file_name + ".jpg")

        xml_file_info = open(xml_file, encoding='utf-8')
        tree = ET.parse(xml_file_info)
        root = tree.getroot()

        for obj in root.iter('object'):
            cls = obj.find('name').text
            box = obj.find('bndbox')

            if cls == "person":
                num += 1

                xmin = int(box.find('xmin').text)
                xmax = int(box.find('xmax').text)
                ymin = int(box.find('ymin').text)
                ymax = int(box.find('ymax').text)

                w = xmax - xmin
                h = ymax - ymin

                # if w >= 80 and h >= 160:
                cut_image_save(img_file, save_dir, data_type, num, xmin, ymin, w, h)
