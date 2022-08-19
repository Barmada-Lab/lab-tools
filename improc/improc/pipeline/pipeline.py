from improc.pipeline.node import Node


class Pipeline:

    def __init__(self, nodes: list[Node]) -> None:
        self.nodes = nodes

    def run(self, experiment) -> None:
        for node in self.nodes:
            experiment = node.transform(experiment)
