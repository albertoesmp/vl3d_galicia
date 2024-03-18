# ----------------------------------------------------------------------------
# AUTHOR: Alberto M. Esmoris Pena
# BRIEF: Data related to the class systems involved in the project
# ----------------------------------------------------------------------------

# ------------------------------------- #
# ---   C L A S S   S Y S T E M S   --- #
# ------------------------------------- #
# Default PNOA-II classes
pnoa2 = {
    'unclassified': 1,
    'ground': 2,
    'lowveg': 3,
    'midveg': 4,
    'highveg': 5,
    'building': 6,
    'noise': 7,
    'water': 9,
    'overlap': 12,
    'bridge': 17
}

# The different classifications tasks
CLASSIF_TYPES = [
    'ORIGINAL', 'VEGETATION', 'LMH_VEGETATION', 'BUILDING', 'BUILD_VEG'
]

# The class names in the database as values (keys are assoc. internal names)
CLASS_NAMES = {
    'unclassified': 'Unclassified',
    'ground': 'Ground',
    'lowveg': 'Low vegetation',
    'midveg': 'Mid vegetation',
    'highveg': 'High vegetation',
    'building': 'Building',
    'noise': 'Noise',
    'water': 'Water',
    'overlap': 'Overlap',
    'bridge': 'Bridge',
    'vegetation': 'Vegetation',
    'other': 'Other',
    'ignore': 'Ignore'
}
