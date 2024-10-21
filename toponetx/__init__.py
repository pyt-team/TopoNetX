"""Initialize the library with modules and other content."""

# From the following modules, we do not import all symbols, but only the module itself.
import toponetx.datasets as datasets

# From the following modules, we import all symbols.
from toponetx.algorithms import *
from toponetx.classes import *
from toponetx.exception import *
from toponetx.generators import *
from toponetx.readwrite import *
from toponetx.transform import *
from toponetx.utils import *

__all__ = ["datasets"]
