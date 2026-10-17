"""CWRU Bearing Data Center files used here, read from the official 12k drive-end tables.

Source: https://engineering.case.edu/bearingdatacenter (Case Western Reserve University).
Only the complete, balanced subset is used: normal baseline, and inner race (IR), ball (B)
and outer race centred at 6:00 (OR) faults at 0.007, 0.014 and 0.021 inch diameters, each
at motor loads 0-3 hp. Other positions and the 0.028 inch faults are incomplete and skipped.
"""

URL = "https://engineering.case.edu/sites/default/files/{n}.mat"

# (fault class, fault diameter in inches, motor load in hp, file number, label on the official page)
FILES = [
    ('B', 0.007, 0, 118, 'B007_0'),
    ('B', 0.007, 1, 119, 'B007_1'),
    ('B', 0.007, 2, 120, 'B007_2'),
    ('B', 0.007, 3, 121, 'B007_3'),
    ('B', 0.014, 0, 185, 'B014_0'),
    ('B', 0.014, 1, 186, 'B014_1'),
    ('B', 0.014, 2, 187, 'B014_2'),
    ('B', 0.014, 3, 188, 'B014_3'),
    ('B', 0.021, 0, 222, 'B021_0'),
    ('B', 0.021, 1, 223, 'B021_1'),
    ('B', 0.021, 2, 224, 'B021_2'),
    ('B', 0.021, 3, 225, 'B021_3'),
    ('IR', 0.007, 0, 105, 'IR007_0'),
    ('IR', 0.007, 1, 106, 'IR007_1'),
    ('IR', 0.007, 2, 107, 'IR007_2'),
    ('IR', 0.007, 3, 108, 'IR007_3'),
    ('IR', 0.014, 0, 169, 'IR014_0'),
    ('IR', 0.014, 1, 170, 'IR014_1'),
    ('IR', 0.014, 2, 171, 'IR014_2'),
    ('IR', 0.014, 3, 172, 'IR014_3'),
    ('IR', 0.021, 0, 209, 'IR021_0'),
    ('IR', 0.021, 1, 210, 'IR021_1'),
    ('IR', 0.021, 2, 211, 'IR021_2'),
    ('IR', 0.021, 3, 212, 'IR021_3'),
    ('N', 0.0, 0, 97, 'Normal_0'),
    ('N', 0.0, 1, 98, 'Normal_1'),
    ('N', 0.0, 2, 99, 'Normal_2'),
    ('N', 0.0, 3, 100, 'Normal_3'),
    ('OR', 0.007, 0, 130, 'OR007@6_0'),
    ('OR', 0.007, 1, 131, 'OR007@6_1'),
    ('OR', 0.007, 2, 132, 'OR007@6_2'),
    ('OR', 0.007, 3, 133, 'OR007@6_3'),
    ('OR', 0.014, 0, 197, 'OR014@6_0'),
    ('OR', 0.014, 1, 198, 'OR014@6_1'),
    ('OR', 0.014, 2, 199, 'OR014@6_2'),
    ('OR', 0.014, 3, 200, 'OR014@6_3'),
    ('OR', 0.021, 0, 234, 'OR021@6_0'),
    ('OR', 0.021, 1, 235, 'OR021@6_1'),
    ('OR', 0.021, 2, 236, 'OR021@6_2'),
    ('OR', 0.021, 3, 237, 'OR021@6_3'),
]


# LOCKED confirmatory set. Never used for training, tuning, or choosing features or models;
# evaluated once per variant at the end. Each row: (fault class, diameter in inches, load hp,
# file number, label, why it is out of distribution).
LOCKED_FILES = [
    ('B', 0.028, 0, 3005, 'B028_0', 'diameter 0.028 (larger than any training fault)'),
    ('B', 0.028, 1, 3006, 'B028_1', 'diameter 0.028 (larger than any training fault)'),
    ('B', 0.028, 2, 3007, 'B028_2', 'diameter 0.028 (larger than any training fault)'),
    ('B', 0.028, 3, 3008, 'B028_3', 'diameter 0.028 (larger than any training fault)'),
    ('IR', 0.028, 0, 3001, 'IR028_0', 'diameter 0.028 (larger than any training fault)'),
    ('IR', 0.028, 1, 3002, 'IR028_1', 'diameter 0.028 (larger than any training fault)'),
    ('IR', 0.028, 2, 3003, 'IR028_2', 'diameter 0.028 (larger than any training fault)'),
    ('IR', 0.028, 3, 3004, 'IR028_3', 'diameter 0.028 (larger than any training fault)'),
    ('OR', 0.007, 0, 156, 'OR007@12_0', 'outer race at 12:00 (not the 6:00 training position)'),
    ('OR', 0.007, 1, 158, 'OR007@12_1', 'outer race at 12:00 (not the 6:00 training position)'),
    ('OR', 0.007, 2, 159, 'OR007@12_2', 'outer race at 12:00 (not the 6:00 training position)'),
    ('OR', 0.007, 3, 160, 'OR007@12_3', 'outer race at 12:00 (not the 6:00 training position)'),
    ('OR', 0.007, 0, 144, 'OR007@3_0', 'outer race at 3:00 (not the 6:00 training position)'),
    ('OR', 0.007, 1, 145, 'OR007@3_1', 'outer race at 3:00 (not the 6:00 training position)'),
    ('OR', 0.007, 2, 146, 'OR007@3_2', 'outer race at 3:00 (not the 6:00 training position)'),
    ('OR', 0.007, 3, 147, 'OR007@3_3', 'outer race at 3:00 (not the 6:00 training position)'),
    ('OR', 0.021, 0, 258, 'OR021@12_0', 'outer race at 12:00 (not the 6:00 training position)'),
    ('OR', 0.021, 1, 259, 'OR021@12_1', 'outer race at 12:00 (not the 6:00 training position)'),
    ('OR', 0.021, 2, 260, 'OR021@12_2', 'outer race at 12:00 (not the 6:00 training position)'),
    ('OR', 0.021, 3, 261, 'OR021@12_3', 'outer race at 12:00 (not the 6:00 training position)'),
    ('OR', 0.021, 0, 246, 'OR021@3_0', 'outer race at 3:00 (not the 6:00 training position)'),
    ('OR', 0.021, 1, 247, 'OR021@3_1', 'outer race at 3:00 (not the 6:00 training position)'),
    ('OR', 0.021, 2, 248, 'OR021@3_2', 'outer race at 3:00 (not the 6:00 training position)'),
    ('OR', 0.021, 3, 249, 'OR021@3_3', 'outer race at 3:00 (not the 6:00 training position)'),
]
