from typing import Protocol, TypeVar, cast

from frontend.ast.node import Node, NullType
from frontend.ast.tree import *
from frontend.ast.visitor import RecursiveVisitor, Visitor
from frontend.scope.globalscope import GlobalScope
from frontend.scope.scope import Scope, ScopeKind
from frontend.scope.scopestack import ScopeStack
from frontend.symbol.funcsymbol import FuncSymbol
from frontend.symbol.symbol import Symbol
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from frontend.type.type import DecafType
from utils.error import *
from utils.riscv import MAX_INT

"""
The namer phase: resolve all symbols defined in the abstract syntax tree and store them in symbol tables (i.e. scopes).
"""


class Namer(Visitor[ScopeStack, None]):
    def __init__(self) -> None:
        pass

    # Entry of this phase
    def transform(self, program: Program) -> Program:
        # Global scope. You don't have to consider it until Step 9.
        program.globalScope = GlobalScope
        ctx = ScopeStack(program.globalScope)

        program.accept(self, ctx)
        return program

    def visitProgram(self, program: Program, ctx: ScopeStack) -> None:
        # Check if the 'main' function is missing
        if not program.hasMainFunc():
            raise DecafNoMainFuncError

        for child in program:
            child.accept(self, ctx)

        func_symbol = FuncSymbol("fill_n", False, INT, Scope(ScopeKind.GLOBAL))
        func_symbol.addParaType(ArrayType(INT, 0))
        func_symbol.addParaType(INT)
        func_symbol.addParaType(INT)
        ctx.globalscope.declare(func_symbol)

    def visitParameter(self, param: Parameter, ctx: ScopeStack) -> None:
        if ctx.findConflict(param.ident.value) is None:
            symbol = VarSymbol(param.ident.value, param.var_type.type)
            ctx.declare(symbol)
            param.setattr("symbol", symbol)
        else:
            raise DecafDeclConflictError(param.ident.value)

    def visitCall(self, call: Call, ctx: ScopeStack) -> None:
        if not ctx.globalscope.containsKey(call.ident.value):
            raise DecafUndefinedFuncError(call.ident.value)
        if not ctx.lookup(call.ident.value).isFunc:
            raise DecafBadFuncCallError(call.ident.value)
        func_symbol = ctx.globalscope.get(call.ident.value)
        if len(func_symbol.para_type) != len(call.parameters):
            raise DecafBadFuncCallError(call.ident.value)
        call.setattr("symbol", func_symbol)
        for param in call.parameters:
            param.accept(self, ctx)

    def visitFunction(self, func: Function, ctx: ScopeStack) -> None:
        func_defined = True if func.body else False
        if ctx.globalscope.containsKey(func.ident.value):
            func_symbol: FuncSymbol = ctx.globalscope.get(func.ident.value)
            if func_symbol.defined and func_defined:
                raise DecafDeclConflictError(func.ident.value)
            else:
                func_symbol.defined = func_symbol.defined or func_defined
        else:
            func_symbol = FuncSymbol(func.ident.value, func_defined, func.ret_t.type, Scope(ScopeKind.GLOBAL))
            ctx.globalscope.declare(func_symbol)
        ctx.open(Scope(ScopeKind.LOCAL))
        for param in func.parameters:
            param.accept(self, ctx)
            func_symbol.addParaType(param.var_type.type)
        if func.body:
            for child in func.body:
                child.accept(self, ctx)
        func.setattr('symbol', func_symbol)
        ctx.close()
        
    def visitBlock(self, block: Block, ctx: ScopeStack) -> None:
        ctx.open(Scope(ScopeKind.LOCAL))
        for child in block:
            child.accept(self, ctx)
        ctx.close()

    def visitReturn(self, stmt: Return, ctx: ScopeStack) -> None:
        stmt.expr.accept(self, ctx)

    def visitFor(self, stmt: For, ctx: ScopeStack) -> None:
        """
        1. Open a local scope for stmt.init.
        2. Visit stmt.init, stmt.cond, stmt.update.
        3. Open a loop in ctx (for validity checking of break/continue)
        4. Visit body of the loop.
        5. Close the loop and the local scope.
        """
        ctx.open(Scope(ScopeKind.LOCAL))
        stmt.init.accept(self, ctx)
        stmt.ctrl.accept(self, ctx)
        stmt.post.accept(self, ctx)
        ctx.openLoop()
        stmt.body.accept(self, ctx)
        ctx.closeLoop()
        ctx.close()

    def visitIf(self, stmt: If, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        stmt.then.accept(self, ctx)

        # check if the else branch exists
        if not stmt.otherwise is NULL:
            stmt.otherwise.accept(self, ctx)

    def visitWhile(self, stmt: While, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        ctx.openLoop()
        stmt.body.accept(self, ctx)
        ctx.closeLoop()

        
    def visitDoWhile(self, stmt: DoWhile, ctx: ScopeStack) -> None:
        """
        1. Open a loop in ctx (for validity checking of break/continue)
        2. Visit body of the loop.
        3. Close the loop.
        4. Visit the condition of the loop.
        """
        ctx.openLoop()
        stmt.body.accept(self, ctx)
        ctx.closeLoop()
        stmt.cond.accept(self, ctx)

    def visitBreak(self, stmt: Break, ctx: ScopeStack) -> None:
        if not ctx.inLoop():
            raise DecafBreakOutsideLoopError()

    def visitContinue(self, stmt: Continue, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBreak.
        """
        if not ctx.inLoop():
            raise DecafContinueOutsideLoopError()

    def visitDeclaration(self, decl: Declaration, ctx: ScopeStack) -> None:
        """
        1. Use ctx.findConflict to find if a variable with the same name has been declared.
        2. If not, build a new VarSymbol, and put it into the current scope using ctx.declare.
        3. Set the 'symbol' attribute of decl.
        4. If there is an initial value, visit it.
        """
        if ctx.isGlobalScope():
            var_defined = True if decl.init_expr else False
            if ctx.globalscope.containsKey(decl.ident.value):
                global_symbol: VarSymbol = ctx.globalscope.get(decl.ident.value)
                if global_symbol.defined and var_defined:
                    raise DecafGlobalVarDefinedTwiceError(decl.ident.value)
                else:
                    return
            else:
                for size in decl.array_size:
                    if size <= 0:
                        raise DecafBadArraySizeError()
                global_symbol = VarSymbol(decl.ident.value, ArrayType.multidim(decl.var_t.type, *(decl.array_size)), True)
            if decl.init_expr:
                if not isinstance(decl.init_expr, IntLiteral):
                    raise DecafGlobalVarBadInitValueError(decl.ident.value)
                global_symbol.setInitValue(decl.init_expr.value)
            else:
                if isinstance(global_symbol.type, ArrayType):
                    global_symbol.setInitValue(
                        [x.value for x in decl.array_init] if decl.array_init is not None else None
                    )
                else:
                    global_symbol.setInitValue(0)
            ctx.globalscope.declare(global_symbol)
            decl.setattr("symbol", global_symbol)
        else:
            if ctx.findConflict(decl.ident.value) is None:
                for size in decl.array_size:
                    if size <= 0:
                        raise DecafBadArraySizeError()
                symbol = VarSymbol(decl.ident.value, ArrayType.multidim(decl.var_t.type, *(decl.array_size)))
                ctx.declare(symbol)
                decl.setattr("symbol", symbol)
                if decl.init_expr:
                    decl.init_expr.accept(self, ctx)
                else:
                    if isinstance(symbol.type, ArrayType):
                        symbol.setInitValue(
                            [x.value for x in decl.array_init] if decl.array_init is not NULL else None
                        )
                    else:
                        symbol.setInitValue(0)
            else:
                raise DecafDeclConflictError(decl.ident.value)

    def visitAssignment(self, expr: Assignment, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        self.visitBinary(expr, ctx)

    def visitUnary(self, expr: Unary, ctx: ScopeStack) -> None:
        expr.operand.accept(self, ctx)

    def visitBinary(self, expr: Binary, ctx: ScopeStack) -> None:
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)

    def visitCondExpr(self, expr: ConditionExpression, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.cond.accept(self, ctx)
        expr.then.accept(self, ctx)
        expr.otherwise.accept(self, ctx)

    def visitIdentifier(self, ident: Identifier, ctx: ScopeStack) -> None:
        """
        1. Use ctx.lookup to find the symbol corresponding to ident.
        2. If it has not been declared, raise a DecafUndefinedVarError.
        3. Set the 'symbol' attribute of ident.
        """
        symbol = ctx.lookup(ident.value)
        if symbol is None:
            raise DecafUndefinedVarError(ident.value)
        ident.type = symbol.type
        ident.setattr("symbol", symbol)

    def visitArrayCall(self, call: ArrayCall, ctx: ScopeStack) -> None:
        call.array.accept(self, ctx)
        if call.index:
            call.type = call.array.type.base
            call.index.accept(self, ctx)
        else:
            call.type = call.array.type
        call.setattr("symbol", call.array.getattr("symbol"))

    def visitIntLiteral(self, expr: IntLiteral, ctx: ScopeStack) -> None:
        value = expr.value
        if value > MAX_INT:
            raise DecafBadIntValueError(value)
