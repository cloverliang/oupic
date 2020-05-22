from typing import Optional


class Node:
    def __init__(self,
                 dist_to_parent: float,
                 name: str,
                 left_child: Optional["Node"] = None,
                 right_child: Optional["Node"] = None):
        self.dist_to_parent = dist_to_parent
        self.name = name
        self.left_child = left_child
        self.right_child = right_child

    def num_child(self):
        return (self.left_child is not None) + (self.right_child is not None)


class Tree:
    def __init__(self, root: None):
        self._root = root
        self.validate()

    def post_order_traverse(self):
        for nd in self._post_order_traverse(self._root):
            yield nd

    def _post_order_traverse(self, node: Node):
        if node is not None:
            for nd in self._post_order_traverse(node.left_child):
                yield nd
            for nd in self._post_order_traverse(node.right_child):
                yield nd
            yield node

    def validate(self) -> None:
        if not self._validate_node(self._root):
            raise ValueError("Some node has one child.")

    def _validate_node(self, node: Node) -> bool:
        if node is None:
            return True

        n_child = node.num_child
        if n_child == 1:
            return False

        return self._validate_node(node.left_child) and self._validate_node(
            node.right_child)
