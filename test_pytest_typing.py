import pytest
import typing
from typing import Dict, Any, List

def test_typing_imports():
    print("Typing module imported successfully")
    print(f"Dict: {Dict}")
    print(f"Any: {Any}")
    print(f"List: {List}")
    assert Dict is typing.Dict
    assert Any is typing.Any
    assert List is typing.List