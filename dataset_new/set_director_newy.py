from glob import glob
import os.path
import xml.etree.ElementTree as ET


images_dir_train = os.path.abspath("/.dataset_new/train/images")

for xml_path in glob("./dataset_new/train/annotations/*.xml"):
    tree = ET.parse(xml_path)
    filename = tree.find("filename").text
    image_path = os.path.join(images_dir_train, filename)
    tree.find("path").text = image_path
    tree.write(xml_path, encoding="UTF-8")
images_dir_valid = os.path.abspath("/.dataset_new/valid/images")

for xml_path in glob("./dataset_new/valid/annotations/*.xml"):
    tree = ET.parse(xml_path)
    filename = tree.find("filename").text
    image_path = os.path.join(images_dir_valid, filename)
    tree.find("path").text = image_path
    tree.write(xml_path, encoding="UTF-8")