from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterator


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
class DB_DataSchema:
    table: str 
    data_attributes: Dict[str, Any] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    sample_metadata: Optional[Dict[str, Any]] = None
    reagents_metadata: Optional[Dict[str, Any]] = None
    accquisition_metadata: Optional[Dict[str, Any]] = None

    def flattened_specs(self) -> Dict[str, Any]:
        """Return a single dict with all the non-None sub-dicts merged."""
        result: Dict[str, Any] = {}
        for d in (
            self.data_attributes,
            self.quality_metrics,
            self.sample_metadata,
            self.reagents_metadata,
            self.accquisition_metadata,
        ):
            if d:
                result.update(d)
        return result

    def iter_specs(self) -> Iterator[Tuple[str, Any]]:
        """Yield (column_name, spec) for every entry in the flattened spec."""
        for key, val in self.flattened_specs().items():
            yield key, val
    
    
@dataclass
class DB_LabelSchema:
    table: str                                
    label_metadata: Optional[Dict[str, Any]] = None

    def flattened_specs(self) -> Dict[str, Any]:
        """For labels, just merge its single metadata dict."""
        return dict(self.label_metadata) if self.label_metadata else {}

    def iter_specs(self) -> Iterator[Tuple[str, Any]]:
        """Yield (column, spec) pairs from label_metadata."""
        for k, v in self.flattened_specs().items():
            yield k, v