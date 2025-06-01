import os

POINTS_PER_SCAN = 131072
FEATURES_PER_POINT = 5

SAM_DYNAMIC_LABEL_NAMES = [
    "unlabeled",
    "pedestrian",
    "vehicle",
    "bicycle",
    "motorcycle",
    "scooter"
]

SAM_DYNAMIC_LABEL_MAP = {
    "unlabeled": 0,
    "pedestrian": 1,
    "vehicle": 2,
    "bicycle": 3,
    "motorcycle": 4,
    "scooter": 5
}

SAM_DYNAMIC_COLOR_MAP = [
    [0, 0, 0],              # 0 Unknown
    [7, 33, 229],           # 1 Person  (Bright Red)
    [140, 51, 147],         # 2 Car     (Violet)
    [66, 21, 72],           # 3 Bike    (Dark Magenta)
    [67, 31, 116],          # 4 Motorcycle (Indigo)
    [239, 92, 215]          # 5 Scooter (Pinkish)
]

SEM_LABEL_MAP = [
    0,  # "unlabeled"
    1,  # concrete
    2,  # grass
    3,  # rocks
    4,  # speedway bricks
    5,  # red bricks
    6,  # pebble pavement
    7,  # light marbiling tiling
    8,  # dark marble tiling
    9,  # dirt paths
    10,  # road pavement
    11,  # short vegetation
    12,  # porcelain tile
    13,  # metal grates
    14,  # blond marble tiling
    15,  # wood panels
    16,  # patterned tile
    17,  # carpet
    18,  # crosswalk
    19,  # dome mat
    20,  # stairs
    21,  # door mat
    22,  # threshold
    23,  # metal floor
    24,  # unknown
]

SEM_LABEL_REMAP = [
    0,  # "unlabeled"
    1,  # concrete
    2,  # grass
    3,  # rocks
    4,  # speedway bricks
    5,  # red bricks
    6,  # pebble pavement
    7,  # light marbiling tiling -> tiling
    7,  # dark marble tiling -> tiling
    8,  # dirt paths
    9,  # road pavement
    10,  # short vegetation
    7,  # porcelain tile -> tiling
    11,  # metal grates
    7,  # blond marble tiling -> tiling
    12,  # wood panels
    7,  # pattened tile -> tiling
    13,  # carpet
    14,  # crosswalk
    15,  # dome mat -> mat
    16,  # stairs
    15,  # door mat -> mat
    17,  # threshold -> other
    17,  # metal floor -> other
    17,  # unknown -> other
]

SEM_LABEL_CLASS_NAMES = [
    "unlabeled",
    "concrete",
    "grass",
    "rocks",
    "speedway bricks",
    "red bricks",
    "pebble pavement",
    "light marbiling tiling",
    "dark marble tiling",
    "dirt paths",
    "road pavement",
    "short vegetation",
    "porcelain tile",
    "metal grates",
    "blond marble tiling",
    "wood panels",
    "patterned tile",
    "carpet",
    "crosswalk",
    "dome mat",
    "stairs",
    "door mat",
    "threshold",
    "metal floor",
    "other"
]

SEM_LABEL_REMAP_CLASS_NAMES = [
    "unlabeled",
    "concrete",
    "grass",
    "rocks",
    "speedway bricks",
    "red bricks",
    "pebble pavement",
    "tiling",
    "dirt paths",
    "road pavement",
    "short vegetation",
    "metal grates",
    "wood panels",
    "carpet",
    "crosswalk",
    "mat",
    "stairs",
    "other"
]

REMAP_SEM_ID_TO_COLOR = [
    [0, 0, 0],              # 0 Unknown
    [47, 171, 97],          # 1 Concrete
    [200, 77, 159],        # 2 Grass
    [126, 49, 141],          # 3 Rocks
    [55, 128, 235],         # 4 Speedway Bricks
    [8, 149, 174],         # 5 Red Bricks
    [141, 3, 98],        # 6 Pebble Pavement
    # 7 Light Marble Tiling, Dark Marble Tiling, Porcelain Tile, Blond Marble Tiling, Patterned Tile
    [203, 110, 74],
    [78, 57, 127],         # 9 Dirt Paths
    [60, 143, 142],          # 10 Road Pavement
    [187, 187, 17],        # 11 Short Vegetation
    [89, 183, 27],         # 13 Metal Grates
    [150, 81, 244],        # 15 Wood Panel
    [60, 100, 116],         # 17 Carpet
    [156, 207, 153],         # 18 Crosswalk
    [135, 138, 159],        # 19 Dome Mat, Door Mat
    [44, 217, 131],             # 20 Stairs
    [115, 226, 101],            # 22 Threshold, Metal Floor, Unknown
]

SEM_ID_TO_COLOR = [
    [0, 0, 0],              # 0 Unknown
    [47, 171, 97],          # 1 Concrete
    [200, 77, 159],        # 2 Grass
    [126, 49, 141],          # 3 Rocks
    [55, 128, 235],         # 4 Speedway Bricks
    [8, 149, 174],         # 5 Red Bricks
    [141, 3, 98],        # 6 Pebble Pavement
    [203, 110, 74],        # 7 Light Marble Tiling
    [49, 240, 115],          # 8 Dark Marble Tiling
    [78, 57, 127],         # 9 Dirt Paths
    [60, 143, 142],          # 10 Road Pavement
    [187, 187, 17],        # 11 Short Vegetation
    [137, 247, 165],        # 12 Porcelain Tile
    [89, 183, 27],         # 13 Metal Grates
    [134, 29, 80],        # 14 Blond Marble Tiling
    [150, 81, 244],        # 15 Wood Panel
    [163, 77, 159],        # 16 Patterned Tile
    [60, 100, 116],         # 17 Carpet
    [156, 207, 153],         # 18 Crosswalk
    [135, 138, 159],        # 19 Dome Mat
    [44, 217, 131],        # 20 Stairs
    [123, 97, 131],        # 21 Door Mat
    [115, 226, 101],          # 22 Threshold
    [156, 43, 40],          # 23 Metal Floor
    [0, 0, 0]               # 24 Unlabeled
]

OBJ_LABEL_REMAP = [
    0,  # "unlabeled"
    1,  # "car"
    2,  # "pedestrian"
    3,  # "bike"
    3,  # "motorcycle"
    1,  # "golf cart" -> car
    1,  # "truck" -> car
    4,  # "scooter" -> "scooter"
    5,  # "tree" -> "tree"
    6,  # "traffic sign" -> "pole sign"
    7,  # "canopy" -> "canopy"
    8,  # "traffic light" -> "traffic light"
    9,  # "bike rack" -> "bike rack"
    10,  # "bollard" -> "barrier"
    10,  # "construction barrier" -> "barrier"
    11,  # "parking kiosk" -> "kiosk machine"
    12,  # "mailbox" -> "dispenser"
    13,  # "fire hydrant" -> "fire"
    14,  # "freestanding plant" -> "plant"
    15,  # "pole" -> "pole"
    6,  # "informational sign" -> "pole sign"
    16,  # "door" -> "door"
    10,  # "fence" -> "barrier"
    10,  # "railing" -> "barrier"
    17,  # "cone" -> "cone"
    18,  # "chair" -> "chair"
    19,  # "bench" -> "bench"
    20,  # "table" -> "table"
    21,  # "trash can" -> "trash can"
    12,  # "newspaper dispenser" -> "dispenser"
    22,  # "room label" -> "flat sign"
    10,  # "stanchion" -> barrier
    12,  # "sanitizer dispenser" -> "dispenser"
    12,  # "condiment dispenser" -> "dispenser"
    11,  # "vending machine" -> "kiosk machine"
    23,  # "emergency aid kit" -> "aid kit"
    13,  # "fire extinguisher" -> "fire"
    24,  # "computer" -> "electronics"
    24,  # "television" -> "electronics"
    25,  # "other" -> "other"
    25,  # "horse" -> "other"
    1,   # "pickup truck" -> car
    1,   # "delivery truck" -> car
    1,   # "service vehicle" -> car
    1,   # "utility vehicle" -> car
    13,  # "fire alarm" -> "fire"
    11,  # "ATM" -> "kiosk machine"
    26,  # "cart" -> "cart"
    27,  # "couch" -> "couch"
    28,  # "traffic arm" -> "traffic arm"
    22,  # "wall sign" -> "flat sign"
    22,  # "floor sign" -> "flat sign"
    29,  # "door switch" -> "door switch"
    30,  # "emergency phone" -> "phone"
    31,  # "dumpster" -> "dumpster"
    25,  # "vacuum cleaner" -> "other"
    4,   # "segway" -> "scooter"
    1,   # "bus" -> car
    4,   # "skateboard" -> "scooter"
    25  # "water fountain" -> "other"
]

OBJ_LABEL_REMAP_CLASS_NAMES = [
    'Unlabeled',
    'Car',
    'Pedestrian',
    'Bike',
    "Scooter",
    "Tree",
    "Pole Sign",
    "Canopy",
    "Traffic Light",
    "Bike Rack",
    "Barrier",
    "Kiosk Machine",
    "Dispenser",
    "Fire",
    "Plant",
    "Pole",
    "Door",
    "Cone",
    "Chair",
    "Bench",
    "Table",
    "Trash Can",
    "Flat Sign",
    "Aid Kit",
    "Electronics",
    "Other",
    "Cart",
    "Couch",
    "Traffic Arm",
    "Door Switch",
    "Phone",
    "Dumpster"
]

OBJ_LABEL_NAMES = [
    # Dynamic Classes
    "Unlabeled",
    "Car",
    "Pedestrian",
    "Bike",
    "Motorcycle",
    "Golf Cart",  # Unused
    "Truck",  # Unused
    "Scooter",
    # Static Classes
    "Tree",
    "Traffic Sign",
    "Canopy",
    "Traffic Light",
    "Bike Rack",
    "Bollard",
    "Construction Barrier",  # Unused
    "Parking Kiosk",
    "Mailbox",
    "Fire Hydrant",
    # Static Class Mixed
    "Freestanding Plant",
    "Pole",
    "Informational Sign",
    "Door",
    "Fence",
    "Railing",
    "Cone",
    "Chair",
    "Bench",
    "Table",
    "Trash Can",
    "Newspaper Dispenser",
    # Static Classes Indoor
    "Room Label",
    "Stanchion",
    "Sanitizer Dispenser",
    "Condiment Dispenser",
    "Vending Machine",
    "Emergency Aid Kit",
    "Fire Extinguisher",
    "Computer",
    "Television",  # unused
    "Other",
    "Horse",
    # New Classes
    "Pickup Truck",
    "Delivery Truck",
    "Service Vehicle",
    "Utility Vehicle",
    "Fire Alarm",
    "ATM",
    "Cart",
    "Couch",
    "Traffic Arm",
    "Wall Sign",
    "Floor Sign",
    "Door Switch",
    "Emergency Phone",
    "Dumpster",
    "Vacuum Cleaner",  # unused
    "Segway",
    "Bus",
    "Skateboard",
    "Water Fountain"
]

# Generated using OBJ_ID_TO_COLOR and OBJ_LABEL_REMAP
REMAP_OBJ_ID_TO_COLOR = [
    [0,   0,   0],
    [140,  51, 147],
    [7,  33, 229],
    [66,  21,  72],
    [67,  31, 116],
    [159, 137, 254],
    [52,  32, 130],
    [239,  92, 215],
    [4, 108,  69],
    [160, 129,   2],
    [160,  93,   2],
    [254, 145,  38],
    [227, 189,   1],
    [202,  79,  74],
    [255, 196, 208],
    [166, 240,   4],
    [113, 168,   3],
    [14,  60, 157],
    [41, 159, 115],
    [91,  79,  14],
    [220, 184,  94],
    [202, 159,  41],
    [253, 137, 129],
    [97,  37,  32],
    [91,  31,  39],
    [24,  55,  95],
    [0,  87, 192],
    [31,  70, 142],
    [24,  45,  66],
    [30,  54,  11],
    [247, 148,  90],
    [250, 126, 149]
]

OBJ_ID_TO_COLOR = [
    (0, 0, 0),              # -1 Unlabeled (Black)
    (140, 51, 147),         # 0 Car (Violet)
    (7, 33, 229),           # 1 Person (Bright Red)
    (66, 21, 72),           # 2 Bike (Dark Magenta)
    (67, 31, 116),          # 3 Motorcycle (Indigo)
    (159, 137, 254),        # 4 Golf Cart (Light Purple)
    (52, 32, 130),          # 5 Truck (Purple)
    (239, 92, 215),         # 6 Scooter (Pinkish)
    (4, 108, 69),           # 7 Tree (Dark Green)
    (160, 129, 2),          # 8 Traffic Sign (Gold)
    (160, 93, 2),           # 9 Canopy (Dark Gold)
    (254, 145, 38),         # 10 Traffic Lights (Orange)
    (227, 189, 1),          # 11 Bike Rack (Cyan)
    (202, 79, 74),          # 12 Bollard (Soft Red)
    (255, 196, 208),        # 13 Construction Barrier (Pale Red)
    (166, 240, 4),          # 14 Parking Kiosk (Bright Green)
    (113, 168, 3),          # 15 Mailbox (Green)
    (14, 60, 157),          # 16 Fire Hydrant (Bright Blue)
    (41, 159, 115),         # 17 Freestanding Plant (Sea Green)
    (91, 79, 14),           # 18 Pole (Olive)
    (220, 184, 94),         # 19 Informational Sign (Pale Yellow)
    (202, 159, 41),         # 20 Door (Burnt Orange)
    (253, 137, 129),        # 21 Fence (Salmon)
    (97, 37, 32),           # 22 Railing (Dark Orange)
    (91, 31, 39),           # 23 Cone (Reddish-Brown)
    (24, 55, 95),           # 24 Chair (Dark Blue)
    (0, 87, 192),           # 25 Bench (Bright Blue)
    (31, 70, 142),          # 26 Table (Blue)
    (24, 45, 66),           # 27 Trash Can (Dark Blue)
    (30, 54, 11),           # 28 Newspaper Dispenser (Dark Green)
    (247, 148, 90),         # 29 Room Label (Light Brown)
    (250, 126, 149),        # 30 Stanchion (Pink)
    (70, 106, 19),          # 31 Sanitizer Dispenser (Green)
    (128, 132, 0),          # 32 Condiment Dispenser (Lime Green)
    (152, 163, 0),          # 33 Vending Machine (Light Lime)
    (6, 32, 231),           # 34 Emergency Aid Kit (Bright Red)
    (8, 68, 212),           # 35 Fire Extinguisher (Royal Blue)
    (18, 34, 119),          # 36 Computer (Navy Blue)
    (17, 46, 168),          # 37 Television (Dark Blue)
    (203, 226, 37),         # 38 Other (Bright Yellow)
    (255, 83, 0),           # 39 Horse (Bright Orange)
    (100, 34, 168),         # 40 Pickup Truck (Purple)
    (150, 69, 253),         # 41 Delivery Truck (Vivid Purple)
    (46, 22, 78),           # 42 Service Vehicle (Dark Violet)
    (121, 46, 216),         # 43 Utility Vehicle (Bright Purple)
    (37, 95, 238),          # 44 Fire Alarm (Sky Blue)
    (95, 100, 14),          # 45 ATM (Dark Olive)
    (25, 97, 119),          # 46 Cart (Teal)
    (18, 113, 225),         # 47 Couch (Cobalt Blue)
    (207, 66, 89),          # 48 Traffic Arm (Dark Salmon)
    (215, 80, 2),           # 49 Wall Sign (Bright Orange)
    (161, 125, 16),         # 50 Floor Sign (Mustard)
    (82, 46, 22),           # 51 Door Switch (Brown)
    (28, 42, 65),           # 52 Emergency Phone (Dark Brown)
    (0, 140, 180),          # 53 Dumpster (Cyan)
    (0, 73, 207),           # 54 Vacuum Cleaner (Azure)
    (120, 94, 242),         # 55 Segway (Lavender)
    (35, 28, 79),           # 56 Bus (Deep Purple)
    (56, 30, 178),          # 57 Skateboard (Indigo)
    (48, 49, 20)            # 58 Water Fountain (Dark Olive Green)
]

""" BEGIN MAPPING CONSTANTS"""
METADATA_DIR = 'metadata'
CALIBRATION_DIR = 'calibrations'
POSES_DIR = 'poses'
POSES_SUBDIRS = ['dense_global', 'dense']

POINTCLOUD_DIR = '3d_raw'
POINTCLOUD_SUBDIR = ['os1']

CAMERA_DIR = '2d_rect'
CAMERA_SUBDIRS = ['cam0', 'cam1']

DEPTH_DIR = 'depth'
DEPTH_SUBDIRS = ['cam0', 'cam1']

SEM_LABEL_DIR = '3d_semantic'
SEM_LABEL_SUBDIR = ['os1']

ELEVATION_LABEL_DIR = 'elevation'

SSC_LABEL_DIR = '3d_ssc'
SSC_LABEL_SUBDIR = ['']
SSC_LABEL_SIZE = (256, 256, -1)

SOC_LABEL_DIR = '3d_soc'
SOC_LABEL_SUBDIR = ['']
SOC_LABEL_SIZE = (256, 256, -1)

LFD_LABEL_DIR = 'actions'
LFD_LABEL_SUBDIR = ['']

FSC_LABEL_DIR = '3d_fsc'
FSC_LABEL_SUBDIR = ['']

SAM_LABEL_DIR = '3d_sam'
SAM_LABEL_SUBDIR = ['']

SAM_DYNAMIC_LABEL_DIR = '3d_sam_dynamic'
SAM_DYNAMIC_LABEL_SUBDIR = ['']

TRAVERSE_LABEL_DIR = 'traversability'
TRAVERSE_LABEL_SUBDIR = ['']

DISTILLATION_LABEL_DIR = 'distillation'

COUNTERFACTUAL_LABEL_DIR = 'counterfactuals'
COUNTERFACTUAL_LABEL_SUBDIR = ['']

TASK_TO_LABEL = {
    SAM_LABEL_DIR: f'{SAM_LABEL_DIR}_label',
    SAM_DYNAMIC_LABEL_DIR: f'{SAM_DYNAMIC_LABEL_DIR}_label',
    FSC_LABEL_DIR: f'{FSC_LABEL_DIR}_label',  # Feature semantic completion
    SSC_LABEL_DIR: f'{SSC_LABEL_DIR}_label',
    SOC_LABEL_DIR: f'{SOC_LABEL_DIR}_label',
    ELEVATION_LABEL_DIR: f'{ELEVATION_LABEL_DIR}_label',
    LFD_LABEL_DIR: f'{LFD_LABEL_DIR}_label',
    TRAVERSE_LABEL_DIR: f'{TRAVERSE_LABEL_DIR}_label',
    COUNTERFACTUAL_LABEL_DIR: f'{COUNTERFACTUAL_LABEL_DIR}_label'
}

LABEL_TO_TASK = {v: k for k, v in TASK_TO_LABEL.items()}

LABEL_TO_ID = {
    "3d_ssc_label": [SEM_LABEL_MAP, SEM_LABEL_REMAP],
    "3d_soc_label": [OBJ_LABEL_NAMES, OBJ_LABEL_REMAP],
    "3d_sam_dynamic_label": [SAM_DYNAMIC_LABEL_NAMES, SAM_DYNAMIC_LABEL_MAP],
    "3d_raw_label": [None, None],
    "elevation_label": [None, None],
    "3d_fsc_label": [None, None],
    "counterfactuals_label": [None, None],
    "traversability_label": [None, None]
}

LABEL_TO_MODEL_PREDS = {
    "3d_ssc_label": "inpainting_preds",
    "3d_fsc_label": "inpainting_preds",  # Mutually exclusive with 3d_ssc_label
    "3d_soc_label": "inpainting_object_preds",
    "3d_sam_label": "inpainting_sam_preds",
    "3d_sam_dynamic_label": "inpainting_sam_dynamic_preds",
    "elevation_label": "elevation_preds",
    "actions_labels": "actions_preds",
    "traversability_label": "traversability_preds"
}

ANNOTATION_TYPE_MAP = {
    "SemanticSegmentation": "SemanticSegmentation",
    "SemanticSceneCompletion": "SemanticSegmentation",
    "Depth": "Depth",
    "ObjectTracking": "ObjectTracking"
}

IGNORE_ELEVATION_CLASSES = [
    0,  # "unlabeled"
]

OUSTER_HEIGHT_REL_GROUND = 0.8
OUSTER_HEIGHT_OFFSET_GROUND = -OUSTER_HEIGHT_REL_GROUND + 0.3
OUSTER_CLOUD_DIM = (131072, 4)

""" END MAPPING CONSTANTS"""


def fn2frame(fn):
    assert isinstance(fn, str), f"Expected string, got {type(fn)}"
    return int(os.path.splitext(os.path.basename(fn))[0].split("_")[-1])


def frame2fn(modality, sensor_name, sequence, frame, filetype):
    sensor_filename = "%s_%s_%s_%s.%s" % (
        modality,
        sensor_name,
        str(sequence),
        str(frame),
        filetype
    )
    return sensor_filename


def fn2info(fn):
    filename_prefix = fn.split('.')[0]
    filename_prefix = filename_prefix.split('_')

    modality = filename_prefix[0]+"_"+filename_prefix[1]
    sensor_name = filename_prefix[2]
    trajectory = filename_prefix[3]
    frame = filename_prefix[4]
    return (modality, sensor_name, trajectory, frame)


def fn2path(root_dir, fn):
    modality, sensor_name, trajectory, frame = fn2info(fn)
    return os.path.join(root_dir, modality, sensor_name, trajectory, fn)
