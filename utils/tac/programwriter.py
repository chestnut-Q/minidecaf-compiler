from __future__ import annotations

from typing import Any, Optional, Union
from frontend.ast.tree import Function

from utils.label.funclabel import *
from utils.label.label import Label, LabelKind
from utils.tac.tacinstr import Global

from .context import Context
from .funcvisitor import FuncVisitor
from .tacprog import TACProg


class ProgramWriter:
    def __init__(self, funcs: dict[str, Function]) -> None:
        self.funcs: list[Function] = []
        self.ctx = Context()
        self.global_vars: list[Global] = []
        for func in funcs.values():
            self.funcs.append(func.ident.value)
            self.ctx.putFuncLabel(func.ident.value, len(func.parameters))

    def visitMainFunc(self) -> FuncVisitor:
        entry = MAIN_LABEL
        return FuncVisitor(entry, 0, self.ctx)

    def visitFunc(self, name: str, numArgs: int) -> FuncVisitor:
        entry = self.ctx.getFuncLabel(name)
        return FuncVisitor(entry, numArgs, self.ctx)

    def visitEnd(self) -> TACProg:
        return TACProg(self.ctx.funcs, self.global_vars)
