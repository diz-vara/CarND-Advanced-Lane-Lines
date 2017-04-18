# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:19:48 2017

@author: diz
"""

images = []
for entry in os.scandir('./test_images'):
    if entry.is_file():
        img = cv2.imread(entry.path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        images.append(img)