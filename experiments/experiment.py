from typing import NamedTuple, List, Dict, Set
from pathlib import Path
import csv
from dataclasses import dataclass

from relationality.fields import Histo, entropy

class MessageEvent(NamedTuple):
    timestamp: int
    source: str
    dest: str


@dataclass
class MessageEventSeries:
    events: List[MessageEvent]
    sources: Set[str]
    dests: Set[str]

    @classmethod
    def from_iterator(cls, it) -> 'MessageEventSeries':
        events = []
        sources = set()
        dests = set()
        for event in it:
            events.append(event)
            sources.add(event.source)
            dests.add(event.dest)
        return cls(sorted(events), sources, dests)
    
    @classmethod
    def from_path(cls, path: Path) -> 'MessageEventSeries':
        with path.open('r') as fh:
            return cls.from_iterator(
                    MessageEvent(int(tsstring), source, dest)
                    for (source, dest, tsstring) in csv.reader(fh))

    def __iter__(self):
        return iter(self.events)


@dataclass
class Node:
    histo: Histo
    entropy: float


class Model:
    series: MessageEventSeries
    nodes: Dict[str, Node]
    dests_list: List[str]
    dests_indices: Dict[str, int]
    entropy: float

    def __init__(self, series: MessageEventSeries):
        # Filter out destinations that are never sources
        # This could be made optional of course
        self.dests_list = list(series.sources & series.dests)
        self.dests_indices = {dest: k for k, dest in enumerate(self.dests_list)}
        self.series = series
        self.reset()

    @classmethod
    def from_path(cls, path: Path) -> 'Model':
        return cls(MessageEventSeries.from_path(path))

    def reset(self):
        dests_count = len(self.dests_list)
        init_entropy = entropy(Histo.uniform(dests_count))
        self.nodes = {
            source: Node(Histo.uniform(dests_count), init_entropy)
            for source in self.series.sources}
        self.entropy = dests_count

    def step(self, event: MessageEvent):
        node = self.nodes.get(event.source)
        if node is None:
            raise ValueError(f"{event.source} has no node")

        old_entropy = node.entropy
        dest_index = self.dests_indices.get(event.dest)
        if dest_index is None:
            return None

        node.histo[dest_index] += 1
        node.entropy = entropy(node.histo)
        delta_entropy = node.entropy - old_entropy
        self.entropy += delta_entropy
        return node


if __name__ == "__main__":
    enron_path = Path("../data/enron.csv")
    model = Model.from_path(enron_path)

    steps = 0
    for ev in model.series:
        node = model.step(ev)
        if node is None:
            continue

        if steps % 20 == 0:
            print(model.entropy)
        steps += 1

    print("Entropies")
