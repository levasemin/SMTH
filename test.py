import Filter
from PIL import Image
import numpy as np

image = np.array(Image.open("british.png"))
kernel = [0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0,
          0, 0, -1, 1, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0]

out = Filter.make_filter(image, kernel, image.shape[1], image.shape[0], 7, 7)

# image = Image.fromarray(out.astype('uint8'))

# image.save("british_py.png")
