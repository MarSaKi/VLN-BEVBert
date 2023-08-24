import numpy as np

# -- crop memories to desired size
def crop_memories(memory, size):
    """

    crop input memory spatially (along first two dimensions) to given size.

    INPUTS:
        memory (3D or 2D np.array): 3rd dim = features dimension
        size (tuple of 2 elements): desired size

    RETURN:
        memory cropped (3D or 2D np.array)

    """

    map_height, map_width = size
    map_height_padded, map_width_padded = memory.shape[:2]

    if len(memory.shape)>2:
        map = ~(memory==0).all(axis=2)
        map = map.astype(np.uint8)
    else:
        map = memory

    # -- x-coord
    map_x = np.sum(map, axis=0)
    #min_x = np.argmax(map_x>0)
    #max_x = len(map_x) - np.argmax(map_x[::-1]>0) - 1

    #center_x = (max_x + min_x) / 2
    center_x = np.round(np.sum(np.multiply(map_x, np.arange(len(map_x)))) /  np.sum(map_x))

    new_min_x = center_x - map_width/2
    new_max_x = center_x + map_width/2

    new_min_x = int(np.round(new_min_x))
    new_max_x = int(np.round(new_max_x))
    if new_max_x - new_min_x + 1 > map_width:
        new_max_x -= (new_max_x - new_min_x + 1) - map_width

    if new_min_x < 0:
        new_min_x = 0
        new_max_x = map_width - 1
    elif new_max_x > map_width_padded - 1:
        new_max_x = map_width_padded - 1
        new_min_x = map_width_padded - map_width

    # -- y-coord
    map_y = np.sum(map, axis=1)
    #min_y = np.argmax(map_y>0)
    #max_y = len(map_y) - np.argmax(map_y[::-1]>0) - 1

    #center_y = (max_y + min_y) / 2
    center_y = np.round(np.sum(np.multiply(map_y, np.arange(len(map_y)))) /  np.sum(map_y))

    new_min_y = center_y - map_height/2
    new_max_y = center_y + map_height/2

    new_min_y = int(np.round(new_min_y))
    new_max_y = int(np.round(new_max_y))
    if new_max_y - new_min_y + 1 > map_height:
        new_max_y -= (new_max_y - new_min_y + 1) - map_height

    if new_min_y < 0:
        new_min_y = 0
        new_max_y = map_height - 1
    elif new_max_y > map_height_padded - 1:
        new_max_y = map_height_padded - 1
        new_min_y = map_height_padded - map_height

    new_min_y = int(np.round(new_min_y))
    new_max_y = int(np.round(new_max_y))

    new_memory = memory[new_min_y:new_max_y+1, new_min_x:new_max_x+1]

    assert new_memory.shape[0] == map_height
    assert new_memory.shape[1] == map_width

    return new_memory, (new_min_y, new_max_y, new_min_x, new_max_x)


