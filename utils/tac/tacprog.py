from utils.tac.tacinstr import Global
from .tacfunc import TACFunc


# A TAC program consists of several TAC functions.
class TACProg:
    def __init__(self, funcs: list[TACFunc], global_vars: list[Global]) -> None:
        self.funcs = funcs
        self.global_vars = global_vars

    def printTo(self) -> None:
        for var in self.global_vars:
            var.printTo()
        for func in self.funcs:
            func.printTo()
