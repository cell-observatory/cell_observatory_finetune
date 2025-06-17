import os
import json
from tqdm import tqdm
from pathlib import Path
from functools import reduce
from itertools import product
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Tuple, Hashable, Sequence,  Dict, Union

import sqlite3
import pandas as pd
from supabase import Client

from cell_observatory_finetune.data.utils import sql_quote_ident
from cell_observatory_finetune.data.databases.schema import TableSchemaList, Comparators


class Database(ABC):

    @abstractmethod
    def _fetch(self, 
               database : str | Path | Client, 
               table: str 
    ):
        """
        Return rows whose columns satisfy the query.
        """
        pass