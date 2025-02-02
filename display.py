from PIL import Image
import numpy as np

def display(data, name):

    w, h = 512, 512
    #data = np.zeros((h, w, 3), dtype=np.uint8)
    #data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
    img = Image.fromarray(data)
    img.save(name + '.png')
    img.show()
