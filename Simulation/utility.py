from colour import Color
import numpy as np

'''
The mid-point circle drawing algorithm returns the points on a discrete 2D map
that should be filled to form a circle of radius r around the point (midx, midy)
'''
def circle_points(midx, midy, r):
    coords = list()
    x = r
    y = 0

    coords.append((x + midx, y + midy))

    # When radius is zero there is only a single point
    if (r > 0):
        coords.append((-x + midx, -y + midy))
        coords.append(( y + midx, -x + midy))
        coords.append((-y + midx,  x + midy))

    # Initialising the value of P
    P = 1 - r
    while (x > y):
        y += 1

        # Mid-points inside or on the perimeter
        if (P <= 0):
            P = P + 2 * y + 1
        # Mid-points outside the perimeter
        else:
            x -= 1
            P = P + 2 * y - 2 * x + 1

        # All the perimeter points have already been printed
        if (x < y):
            break

        # Get the point and its reflection in the other octants
        coords.append(( x + midx,  y + midy))
        coords.append((-x + midx,  y + midy))
        coords.append(( x + midx, -y + midy))
        coords.append((-x + midx, -y + midy))

        # If the generated points on the line x = y then
        # the perimeter points have already been printed
        if (x != y):
            coords.append(( y + midx,  x + midy))
            coords.append((-y + midx,  x + midy))
            coords.append(( y + midx, -x + midy))
            coords.append((-y + midx, -x + midy))

    return coords

# Find the middle point of the map to start the fire
def get_fire_location(width, height):
    midx = int(width / 2)
    midy = int(height / 2)
    return (midx, midy)

def get_agent_location(width, height):
    # Minimum map size is 10x10
    assert width >= 10 and height >= 10
    # Agent distance from fire varies but is never too far away
    radius = np.random.choice([1, 2, 3])
    # Don't draw circles larger than the map
    midx, midy = get_fire_location(width, height)
    locations = circle_points(midx, midy, radius)
    # Choose a random point on the circle
    random_idx = np.random.choice(np.array(range(len(locations))))
    random_loc = locations[random_idx]
    # Convert to int because otherwise not JSON serializable
    return (int(random_loc[0]), int(random_loc[1]))

# Generate a unique name for each run, based on constants and the current time
def get_name(epsiodes, memories, size):
    import time
    return (
        f"""{epsiodes}k-{memories}m-{size}s-VAR_{time.strftime("%m-%d-%H%M")}"""
    )

# Convert a color to grayscale with the grayscale formula from Wikipedia
def grayscale(color):
    r, g, b = color.red, color.green, color.blue
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# The parameters for grass
grass = {
    "gray"         : grayscale(Color("Green")),
    "gray_burning" : grayscale(Color("Red")),
    "gray_burnt"   : grayscale(Color("Black")),
    "heat"      : 0.3,
    "fuel"      : 20,
    "threshold" : 3,
    "radius"    : 1,
}

# The parameters for dirt
dirt = {
    "gray" : grayscale(Color("Brown")),
    "heat" : -1,
    "fuel" : -1,
    "threshold" : -1,
}

# The (depth) layers of the map, which corresponds to cell attributes
layer = {
    "type" : 0,
    "gray" : 1,
    "temp" : 2,
    "heat" : 3,
    "fuel" : 4,
    "threshold" : 5,
    "agent_pos" : 6,
    "fire_mobility" : 7,
}

# Which cell type (from the type layer) corresponds to which value
types = {
    0 : "grass",
    1 : "fire",
    2 : "burnt",
    3 : "road",

    "grass" : 0,
    "fire"  : 1,
    "burnt" : 2,
    "road"  : 3,
}

# Convert grayscale to ascii for rendering
color2ascii = {
    grayscale(Color("Green")) : '+',
    grayscale(Color("Red"))   : '@',
    grayscale(Color("Black")) : '#',
    grayscale(Color("Brown")) : '0',
}
