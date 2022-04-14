import numpy as np
from scipy.spatial.distance import pdist


def fix_height(subset_tree, houppier, radius):
    margin = 0.25
    height_real = np.max(subset_tree[:, 2])

    houppier_calc = None

    if houppier < 4:
        height_to_test = 1.25 * houppier
    elif houppier < 10:
        height_to_test = 1.2 * houppier
    else:
        height_to_test = 2 + houppier

    if height_to_test >= height_real:
        height_to_test = height_real - 1
    while height_to_test - margin > 0.5 * houppier:   #before 0.5
        research_piece = subset_tree[
            (subset_tree[:, 2] >= height_to_test - margin) & (subset_tree[:, 2] < height_to_test)]
        if len(research_piece) > 1:
            width_piece = np.max(pdist(research_piece[:, :2], metric='euclidean'))
            if width_piece < radius * 5:
                houppier_calc = height_to_test
                break
        else:
            houppier_calc = height_to_test
            break

        height_to_test -= margin

    if houppier_calc is None:
        houppier_calc = houppier

    # print(houppier_calc, houppier)
    return houppier_calc