import yaml
from typing import List


class InventoryItem(yaml.YAMLObject):
    position: int = None
    name: str = None
    quantity: float = None
    pose: List[float] = None

    def __init__(self, position: int, name: str, quantity: float, pose: List[float]):
        self.position = position
        self.name = name
        self.quantity = quantity
        self.pose = pose

    def __repr__(self) -> str:
        return f"position: {self.position}, name: {self.name}, quantity: {self.quantity}, pose: {self.pose}"


def inventoryitem_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> InventoryItem:
    """Construct an inventory item."""
    return InventoryItem(**loader.construct_mapping(node))


class Ingredient(yaml.YAMLObject):
    name: str = None
    quantity: float = None

    def __init__(self, name: str, quantity: float):
        self.name = name
        self.quantity = quantity

    def __repr__(self) -> str:
        return f"name: {self.name}, quantity: {self.quantity}"


def ingredient_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> Ingredient:
    """Construct an ingredient."""
    return Ingredient(**loader.construct_mapping(node))


class Recipe(yaml.YAMLObject):
    name: str = None
    id: int = None
    ingredients: List[Ingredient] = None

    def __init__(self, id: int, name: str, ingredients: List[Ingredient]):
        self.id = id
        self.name = name
        self.ingredients = ingredients

    def __repr__(self) -> str:
        return f"name: {self.name}, id: {self.id}, ingredients: {self.ingredients}"


def recipe_constructor(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode) -> Recipe:
    """Construct a recipe."""
    return Recipe(**loader.construct_mapping(node))


def get_yaml_loader():
    """Add constructors to PyYAML loader."""
    loader = yaml.SafeLoader
    loader.add_constructor("!ingredient", ingredient_constructor)
    loader.add_constructor("!recipe", recipe_constructor)
    loader.add_constructor("!inventoryitem", inventoryitem_constructor)
    return loader
