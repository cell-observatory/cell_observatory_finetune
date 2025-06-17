from enum import Enum
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import instantiate
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Iterator


class Comparators(Enum):
    EQUAL                 = "="
    NOT_EQUAL             = "!="
    GREATER_THAN          = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN             = "<"
    LESS_THAN_OR_EQUAL    = "<="
    IN                    = "IN"
    NOT_IN                = "NOT IN"
    LIKE                  = "LIKE"
    BETWEEN               = "BETWEEN"
    IS_NULL               = "IS NULL"
    IS_NOT_NULL           = "IS NOT NULL"


@dataclass
class DB_TableSchema:
    """
    Minimal base table schema.
    """
    table: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def flattened_specs(self) -> Dict[str, Any]:
        """Return a shallow copy of the attributes dict."""
        return dict(self.attributes)

    def iter_specs(self) -> Iterator[Tuple[str, Any]]:
        """Yield (column, spec) pairs one by one."""
        yield from self.attributes.items()


@dataclass
class TableSchemaList:
    """
    Holds an ordered collection of DB_Schema objects with utilities to
    (a) iterate over every (column, spec) in all schemas and
    (b) view the union of all specs as a single flat dict.
    """
    schemas: List[DB_TableSchema] = field(default_factory=list)

    # schemas is a list of node objects hence
    # will not be instantiated by hydra automatically
    def __post_init__(self):
        schemas: List[DB_TableSchema] = []
        for s in self.schemas:
            if isinstance(s, DB_TableSchema):
                schemas.append(s)
            elif isinstance(s, DictConfig):
                if "_target_" in s:
                    schemas.append(instantiate(s))
                else:
                    schemas.append(DB_TableSchema(**s))
            elif isinstance(s, dict):
                schemas.append(DB_TableSchema(**s))
            else:
                raise TypeError(f"Cannot turn {s!r} into a DB_TableSchema")
        self.schemas = schemas

    def __iter__(self):
        return iter(self.schemas)

    def __len__(self):
        return len(self.schemas)