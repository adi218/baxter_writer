
class Digits:

    def __init__(self):
        self.digit = {'0': ((0, -0.5), (1, 0), (0, 0.5), (-1, 0)),
                 '1': [(1, 0)],
                 '2': ((0, 1), (0.5, 0), (0, -1), (0.5, 0), (0, 1)),
                 '3': ((0, -1), (0.5, 0), (0, 0.5), (0, -0.5), (0.5, 0), (0, 1)),
                 '4': ((1, 0), (-0.5, 0), (0, .5), (0.5, 0)),
                 '5': ((0, -0.5), (0.5, 0), (0, 0.5), (0.5, 0), (0, -0.5)),
                 '6': ((0, 0.5), (-1, 0), (0, -0.5), (0.5, 0), (0, 0.5)),
                 '7': ((1, 0), (0, 0.5)),
                 '8': ((0, -0.5), (-0.5, 0), (0, 0.5), (1, 0), (0, 0.5), (-0.5, 0)),
                 '9': ((0, -0.5), (1, 0), (0, 0.5), (-0.5, 0), (0, -0.5))}



