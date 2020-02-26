import numpy as np


class Cube:
    def __init__(self, pos):
        super.__init()
        self.pos = pos
        self.radius = 0.5
        self.verts = np.array(
            [
                self.pos - self.radius,
                self.pos - np.array((-self.radius, self.radius, self.radius)),
                self.pos - np.array((-self.radius, -self.radius, self.radius)),
                self.pos - np.array((self.radius, -self.radius, self.radius)),
                self.pos - np.array((self.radius, self.radius, -self.radius)),
                self.pos - np.array((-self.radius, self.radius, -self.radius)),
                self.pos - np.array((-self.radius, -self.radius, -self.radius)),
                self.pos - np.array((self.radius, -self.radius, -self.radius,)),
            ]
        )
        self.faces = [
            [self.verts[2], self.verts[3], self.verts[7], self.verts[6]],  # back
            [self.verts[0], self.verts[1], self.verts[5], self.verts[4]],  # front
            [self.verts[0], self.verts[1], self.verts[2], self.verts[3]],  # Bottom
            [self.verts[4], self.verts[5], self.verts[6], self.verts[7]],  # top
            [self.verts[4], self.verts[7], self.verts[3], self.verts[0]],  # left
            [self.verts[1], self.verts[2], self.verts[6], self.verts[5]],  # right
        ]
