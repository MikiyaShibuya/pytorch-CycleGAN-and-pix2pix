import numpy as np
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
import joblib




color_palette = [[-21, [0, 0, 0]], [0, [34, 2, 120]], [10, [161, 33, 146]], [25, [224, 120, 46]],
                 [40, [253, 242, 150]], [81, [250, 252, 240]]]
# color_palette = [[-20, [0, 0, 0]], [0, [34, 2, 120]], [80, [250, 252, 240]]]

# color_palette = [[-20, [255,0,0]], [80, [0, 0, 255]]]

temp_range = [-20, 80]
#temp_range = [-10, 40]



def colorize(image, palette, temp_range):

    if len(image.shape) == 3:
        image = image[:,:,0]

    image = (image/255.) * (temp_range[1] - temp_range[0]) + temp_range[0]

    W, H = image.shape
    C = 3

    masks = []
    for i in range(len(palette)):
        masks.append(np.zeros((W, H)))

    for i in range(len(palette) - 1):
        T0 = palette[i][0]
        T1 = palette[i+1][0]

        map0 = np.ones((W, H))
        map1 = np.ones((W, H))

        map0[image < T0] = 0
        map0[image >= T1] = 0

        map1[image < T0] = 0
        map1[image >= T1] = 0

        map0 *= (T1 - image)/(T1 - T0)
        map1 *= (image - T0)/(T1 - T0)

        masks[i] += map0
        masks[i+1] += map1

    out = np.zeros((W, H, C))

    for mask, p in zip(masks, palette):
        color = p[1]

        r = mask * color[0]
        g = mask * color[1]
        b = mask * color[2]
        tmp = np.stack((r, g, b))
        out += tmp.transpose(1, 2, 0)

    i = 0
    #return np.stack((masks[i],masks[i],masks[i])).transpose(1,2,0)
    return out.astype(np.uint8)

def convert(file, image_dn, dest_dir):
    image = np.asarray(Image.open(image_dn))

    out = colorize(image, color_palette, temp_range)

    write_dn = os.path.join(dest_dir, file)

    plt.imsave(write_dn, out)



def main():

    dn = sys.argv[1]
    if not os.path.isdir(dn):
        print("That is not directory.")
        return

    suffix = ""
    if len(sys.argv) >= 3:
        suffix = sys.argv[2]

    files = os.listdir(dn)
    dest_dir = os.path.join(dn, 'thermo')
    os.makedirs(dest_dir, exist_ok=True)

    count = 0

    targets = []

    for file in files:

        image_dn = os.path.join(dn, file)
        if not os.path.isfile(image_dn):
            continue

        if not file.endswith(suffix):
            continue

        if file.startswith('.'):
            continue

        targets.append([file, image_dn])
        count += 1

    #for f, i in targets:
    #    convert(f, i, dest_dir)

    joblib.Parallel(n_jobs=-1)([joblib.delayed(convert)(f, i, dest_dir) for f, i in targets])

    print("{} images are converted.".format(count))

if __name__ == "__main__":
    main()
