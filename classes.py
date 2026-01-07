

class ClassA:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"ClassA(name={self.name}, age={self.age})"