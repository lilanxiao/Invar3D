import os

# ---------------------------------------------------------------------------------------------
# possible path to extracted scannet data (NOT the raw data!)
# please update!
GUESS = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        ]
# ---------------------------------------------------------------------------------------------

TARGET = None
for g in GUESS:
    if os.path.exists(g):
        TARGET = g
        break

if TARGET is None:
    raise IOError("Data not found in following path: ", GUESS)

# some frame name with invalid depth maps (all NaN)
BLACK_LIST = ["scene0243_00/frame-001175",
            "scene0243_00/frame-001176",
            "scene0243_00/frame-001177",
            "scene0243_00/frame-001178",
            "scene0243_00/frame-001179",
            "scene0243_00/frame-001180",
            "scene0243_00/frame-001181",
            "scene0243_00/frame-001182",
            "scene0243_00/frame-001183",
            "scene0243_00/frame-001184", 
            "scene0538_00/frame-001925",
            "scene0538_00/frame-001928",
            "scene0538_00/frame-001929",
            "scene0538_00/frame-001931",
            "scene0538_00/frame-001932",
            "scene0538_00/frame-001933",
            "scene0639_00/frame-000444",
            "scene0639_00/frame-000442",
            "scene0639_00/frame-000443"]