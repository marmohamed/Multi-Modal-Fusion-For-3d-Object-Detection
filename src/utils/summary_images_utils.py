import math
import numpy.matlib as npm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np
import cv2


def get_image(lidar_img, labels):
    plt.clf()
    fig,ax = plt.subplots(1)

    ax.imshow(lidar_img)

    for i in range(0, len(labels)):
        w = labels[i][3] 
        h = labels[i][4] 
        x = labels[i][0]
        y = labels[i][1]
        polygon = convert5Pointto8Point(y, x, w, h, -labels[i][6]*57.2958)
        ax.scatter(y, x, s=10, c='r')
        xs = polygon[0::2]
        ys = polygon[1::2]
        l = []
        for j in range(4):
            l.append([xs[j], ys[j]])

        rect = patches.Polygon(l,linewidth=1.,edgecolor='g',facecolor='none')
        ax.add_patch(rect)

    plot_img_np = get_img_from_fig(fig)

    return plot_img_np



def get_img_from_fig(fig, dpi=90):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def convert5Pointto8Point(cx_, cy_, w_, h_, a_):

    theta = math.radians(a_)
    bbox = npm.repmat([[cx_], [cy_]], 1, 5) + \
            np.matmul([[math.cos(theta), math.sin(theta)],
                        [-math.sin(theta), math.cos(theta)]],
                        [[-w_ / 2, w_/ 2, w_ / 2, -w_ / 2, w_ / 2 + 8],
                        [-h_ / 2, -h_ / 2, h_ / 2, h_ / 2, 0]])
    # add first point
    x1, y1 = bbox[0][0], bbox[1][0]
    # add second point
    x2, y2 = bbox[0][1], bbox[1][1]
    # add third point
    #x3, y3 = bbox[0][4], bbox[1][4]   
    # add forth point
    x3, y3 = bbox[0][2], bbox[1][2]
    # add fifth point
    x4, y4 = bbox[0][3], bbox[1][3]

    return [x1, y1, x2, y2, x3, y3, x4, y4]





