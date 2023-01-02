from typing import Protocol, TypeVar

from frontend.ast.node import Node
from frontend.ast.tree import *
from frontend.ast.visitor import Visitor
from frontend.scope.globalscope import GlobalScope
from frontend.scope.scope import Scope
from frontend.scope.scopestack import ScopeStack
from frontend.type.array import ArrayType
from frontend.symbol.varsymbol import VarSymbol
from frontend.symbol.funcsymbol import FuncSymbol
from utils.error import *

"""
The typer phase: type check abstract syntax tree.
"""


class Typer(Visitor[None, None]):
    def __init__(self) -> None:
        pass

    # Entry of this phase
    def transform(self, program: Program) -> Program:
        program.accept(self, None)
        return program

    def visitProgram(self, program: Program, ctx: None) -> None:
        for child in program:
            child.accept(self, ctx)

    def visitParameter(self, param: Parameter, ctx: None) -> None:
        symbol: VarSymbol = param.getattr("symbol")
        if symbol.type != INT or param.var_type.type != INT:
            raise DecafTypeMismatchError()

    def visitCall(self, call: Call, ctx: None) -> None:
        func_symbol: FuncSymbol = call.getattr("symbol")
        if func_symbol.type != INT:
            raise DecafTypeMismatchError()
        for param in call.parameters:
            param.accept(self, ctx)

    def visitFunction(self, func: Function, ctx: None) -> None:
        func_symbol: FuncSymbol = func.getattr("symbol")
        if func_symbol.type != INT:
            raise DecafTypeMismatchError()
        for param in func.parameters:
            param.accept(self, ctx)
        if func.body:
            for child in func.body:
                child.accept(self, ctx)

    def visitBlock(self, block: Block, ctx: None) -> None:
        for child in block:
            child.accept(self, ctx)

    def visitReturn(self, stmt: Return, ctx: None) -> None:
        stmt.expr.accept(self, ctx)
        if stmt.expr.type != INT:
            raise DecafTypeMismatchError()

    def visitFor(self, stmt: For, ctx: None) -> None:
        stmt.init.accept(self, ctx)
        stmt.ctrl.accept(self, ctx)
        if stmt.ctrl.type != INT:
            raise DecafTypeMismatchError()
        stmt.post.accept(self, ctx)
        stmt.body.accept(self, ctx)

    def visitIf(self, stmt: If, ctx: None) -> None:
        stmt.cond.accept(self, ctx)
        if stmt.cond.type != INT:
            raise DecafTypeMismatchError()
        stmt.then.accept(self, ctx)
        if not stmt.otherwise is NULL:
            stmt.otherwise.accept(self, ctx)

    def visitWhile(self, stmt: While, ctx: None) -> None:
        stmt.cond.accept(self, ctx)
        if stmt.cond.type != INT:
            raise DecafTypeMismatchError()
        stmt.body.accept(self, ctx)
        
    def visitDoWhile(self, stmt: DoWhile, ctx: None) -> None:
        stmt.body.accept(self, ctx)
        stmt.cond.accept(self, ctx)
        if stmt.cond.type != INT:
            raise DecafTypeMismatchError()

    def visitDeclaration(self, decl: Declaration, ctx: None) -> None:
        symbol: VarSymbol = decl.getattr("symbol")
        if isinstance(symbol.type, ArrayType):
            if isinstance(symbol.initValue, int):
                raise DecafTypeMismatchError()
        else:
            if symbol.type != INT:
                raise DecafTypeMismatchError()
        if decl.init_expr:
            decl.init_expr.accept(self, ctx)

    def visitUnary(self, expr: Unary, ctx: None) -> None:
        expr.operand.accept(self, ctx)
        if expr.operand.type != INT:
            raise DecafTypeMismatchError()

    def visitBinary(self, expr: Binary, ctx: None) -> None:
        expr.lhs.accept(self, ctx)
        if expr.lhs.type != INT:
            raise DecafTypeMismatchError()
        expr.rhs.accept(self, ctx)
        if expr.rhs.type != INT:
            raise DecafTypeMismatchError()

    def visitCondExpr(self, expr: ConditionExpression, ctx: None) -> None:
        expr.cond.accept(self, ctx)
        if expr.cond.type != INT:
            raise DecafTypeMismatchError()
        expr.then.accept(self, ctx)
        if expr.then.type != INT:
            raise DecafTypeMismatchError()
        expr.otherwise.accept(self, ctx)
        if expr.otherwise.type != INT:
            raise DecafTypeMismatchError()

    def visitArrayCall(self, call: ArrayCall, ctx: None) -> None:
        call.array.accept(self, ctx)
        if call.index:
            call.index.accept(self, ctx)
            if call.index.type != INT:
                raise DecafTypeMismatchError()

    def visitIntLiteral(self, expr: IntLiteral, ctx: None) -> None:
        if expr.type != INT:
            raise DecafTypeMismatchError()