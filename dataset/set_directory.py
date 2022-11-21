from glob import glob
import os.path
import xml.etree.ElementTree as ET


images_dir = os.path.abspath("images")

for xml_path in glob("annotations/*.xml"):
    tree = ET.parse(xml_path)
    filename = tree.find("filename").text
    image_path = os.path.join(images_dir, filename)
    tree.find("path").text = image_path
    tree.write(xml_path, encoding="UTF-8")
