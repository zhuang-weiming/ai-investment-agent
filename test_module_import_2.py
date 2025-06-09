import pytest
import typing
from src.agents.peter_lynch_agent import PeterLynchAgent

# Define typing aliases after imports
Dict = typing.Dict
Any = typing.Any
List = typing.List

def test_module_imports():
    print("Typing module imported successfully")
    print(f"Dict: {Dict}")
    print(f"Any: {Any}")
    print(f"List: {List}")
    print(f"PeterLynchAgent: {PeterLynchAgent}")
    assert Dict is typing.Dict
    assert Any is typing.Any
    assert List is typing.List
    assert PeterLynchAgent is not None