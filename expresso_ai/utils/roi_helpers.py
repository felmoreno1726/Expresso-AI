import math


def preserve_proportions(rect, dimensions_out):
    """
    Finds the tightest rectangle with the same center as rect such that:
    - rect_out contains the input rectangle 
    - the dimensions of rect_out are proportional to dimensions out
    """
    x, y, w, h = rect
    W, H = dimensions_out
    if W/H > w/h:
        #increase w, fix h
        w_out = int(W * h /H)
        h_out = h
    elif W/H < w/h:
        #fix w, increase h
        w_out = w
        h_out = int(H * w /W)
    else:
        #don't change the proportion
        w_out = w
        h_out = h
    delta_w = w_out - w
    delta_h = h_out - h
    x_out, y_out = (
            max(0, x - delta_w//2), 
            max(0, y - delta_h//2)
    )
    return x_out, y_out, w_out, h_out


def pad_rectangle_box(rect, padding, mode='pixel'):
    """
    bbox: int(tuple) representing the 
    padding: 
    model: (str) 'pixel', 'length', or 'area'. Represents the padding mode to use for the rectangle
    """
    x, y, w, h = rect
    if mode == 'pixel':
        x, y, w, h = (
                max(0, x - padding // 2),
                max(0, y - padding //2),
                w + padding, 
                h + padding
        )
    elif mode == 'length':
        x, y, w, h = (
                max(0, int(x - w * padding / 2)), 
                max(0, int(y - h * padding / 2)),
                int(w + w * padding), 
                int(h + h * padding)
        )
    elif mode == 'area':
        #area = w * h
        #new_area = area + area * padding
        #new_width * new_height = new_area = (area + area * padding) = (w + w*sub_pad) * (h + h*sub_pad) = w*h*(1+subpad)**2
        #(1+subpad)**2 = new_area / (area) = (1 + padding)
        #subpad = sqrt(1 + padding) - 1
        subpad = math.sqrt(1+padding) - 1
        x, y, w, h = (
                max(0, int(x - w * subpad / 2)), 
                max(0, int(y - h * subpad / 2)),
                int(w + w * subpad),
                int(h + h * subpad)
        )
    else:
        raise Exception(f"Mode {mode} is not defined. Valid values are 'pixel', 'length', 'area'")
    return x, y, w, h


