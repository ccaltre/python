class Circle:
    #Class attribute
    pi = 3.14

    #Constructor
    def __init__(self, radius):
        #Instance variable
        self.radius = radius

    def area(self):
        return self.radius**2 * self.pi

