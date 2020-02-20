from QuaternionObject import QuaternionObject
import glm


class Box(QuaternionObject):

    """Docstring for Box. """

    def __init__(self, pos):
        """TODO: to be defined. """
        QuaternionObject.__init__(self)
        self.set_position(pos)
        # Verts of a box centered on (0,0,0) with no rotation.
        self.verts = [
            (-0.5, -0.5, 0),  # bottom left
            (0.5, -0.5, 0),  # bottom right
            (0.5, 0.5, 0),  # top right
            (-0.5, 0.5, 0),  # top left
        ]

    def get_verts(self):
        pos = glm.vec4(self.get_position(), 0)
        # print("Initial relative vertices")
        # print(self.verts)
        model = self.generate_model_matrix()
        rotated_relative = [model * glm.vec4(x, 0) for x in self.verts]
        # print("Rotated relative vertices")
        # print(rotated_relative)
        # print("Adjusted for object's position")
        true_verts = [x + pos for x in rotated_relative]
        # print("Object's origin point: " + str(pos))
        # print("Vertices of the object in world space")
        # print(true_verts)
        return true_verts

