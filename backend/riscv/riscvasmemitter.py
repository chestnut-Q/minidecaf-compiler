from typing import Sequence, Tuple

from backend.asmemitter import AsmEmitter
from utils.error import IllegalArgumentException
from utils.label.label import Label, LabelKind
from utils.riscv import Riscv
from utils.tac.reg import Reg
from utils.tac.tacfunc import TACFunc
from utils.tac.tacinstr import *
from utils.tac.tacvisitor import TACVisitor

from ..subroutineemitter import SubroutineEmitter
from ..subroutineinfo import SubroutineInfo

"""
RiscvAsmEmitter: an AsmEmitter for RiscV
"""


class RiscvAsmEmitter(AsmEmitter):
    def __init__(
        self,
        allocatableRegs: list[Reg],
        callerSaveRegs: list[Reg],
        globalVars: list[Global]
    ) -> None:
        super().__init__(allocatableRegs, callerSaveRegs, globalVars)

    
        # the start of the asm code
        # int step10, you need to add the declaration of global var here
        for var in self.globalVars:
            if var.is_array:
                if var.initialized:
                    self.printer.println(".data")
                    self.printer.println(".global %s" % var.name)
                    self.printer.printLabel(Label(LabelKind.GLOBAL, var.name))
                    init_offset = 0
                    for i in range(len(var.init_value)):
                        self.printer.println(".word %d" % var.init_value[i])
                        init_offset += 4
                    if var.array_size - init_offset > 0:
                        self.printer.println(".zero %d" % (var.array_size - init_offset))
                else:
                    self.printer.println(".bss")
                    self.printer.println(".global %s" % var.name)
                    self.printer.printLabel(Label(LabelKind.GLOBAL, var.name))
                    self.printer.println(".space %d" % var.array_size)
            else:
                if var.initialized:
                    self.printer.println(".data")
                    self.printer.println(".global %s" % var.name)
                    self.printer.printLabel(Label(LabelKind.GLOBAL, var.name))
                    self.printer.println(".word %d" % var.init_value)
                else:
                    self.printer.println(".bss")
                    self.printer.println(".global %s" % var.name)
                    self.printer.printLabel(Label(LabelKind.GLOBAL, var.name))
                    self.printer.println(".space 4")

        self.printer.println(".text")
        self.printer.println(".global main")
        self.printer.println("")

    # transform tac instrs to RiscV instrs
    # collect some info which is saved in SubroutineInfo for SubroutineEmitter
    def selectInstr(self, func: TACFunc) -> tuple[list[str], SubroutineInfo]:

        selector: RiscvAsmEmitter.RiscvInstrSelector = (
            RiscvAsmEmitter.RiscvInstrSelector(func.entry)
        )
        for instr in func.getInstrSeq():
            instr.accept(selector)

        info = SubroutineInfo(func.entry, selector.array_offset)

        return (selector.seq, info)

    # use info to construct a RiscvSubroutineEmitter
    def emitSubroutine(self, info: SubroutineInfo):
        return RiscvSubroutineEmitter(self, info)

    # return all the string stored in asmcodeprinter
    def emitEnd(self):
        return self.printer.close()

    class RiscvInstrSelector(TACVisitor):
        def __init__(self, entry: Label) -> None:
            self.entry = entry
            self.passed_params = []
            self.seq = []
            self.array_offset = 0

        # in step11, you need to think about how to deal with globalTemp in almost all the visit functions. 
        def visitReturn(self, instr: Return) -> None:
            if instr.value is not None:
                self.seq.append(Riscv.Move(Riscv.A0, instr.value))
            else:
                self.seq.append(Riscv.LoadImm(Riscv.A0, 0))
            self.seq.append(Riscv.JumpToEpilogue(self.entry))

        def visitMark(self, instr: Mark) -> None:
            self.seq.append(Riscv.RiscvLabel(instr.label))

        def visitLoadImm4(self, instr: LoadImm4) -> None:
            self.seq.append(Riscv.LoadImm(instr.dst, instr.value))

        def visitLoadSymbol(self, instr: LoadSymbol) -> None:
            self.seq.append(Riscv.LoadSymbol(instr.dst, instr.symbol))

        def visitLoad(self, instr: Load) -> None:
            self.seq.append(Riscv.LoadWord(instr.dst, instr.base, instr.offset))

        def visitStore(self, instr: Store) -> None:
            self.seq.append(Riscv.StoreWord(instr.src, instr.base, instr.offset))

        def visitUnary(self, instr: Unary) -> None:
            self.seq.append(Riscv.Unary(instr.op, instr.dst, instr.operand))
 
        def visitBinary(self, instr: Binary) -> None:
            if instr.op == BinaryOp.EQU:
                self.seq.append(Riscv.Binary(BinaryOp.SUB, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(UnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == BinaryOp.NEQ:
                self.seq.append(Riscv.Binary(BinaryOp.SUB, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(UnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == BinaryOp.LEQ:
                self.seq.append(Riscv.Binary(BinaryOp.SGT, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(UnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == BinaryOp.GEQ:
                self.seq.append(Riscv.Binary(BinaryOp.SLT, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(UnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == BinaryOp.AND:
                self.seq.append(Riscv.Unary(UnaryOp.SNEZ, instr.dst, instr.lhs))
                self.seq.append(Riscv.Unary(UnaryOp.NEG, instr.dst, instr.dst))
                self.seq.append(Riscv.Binary(BinaryOp.AND, instr.dst, instr.dst, instr.rhs))
                self.seq.append(Riscv.Unary(UnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == BinaryOp.OR:
                self.seq.append(Riscv.Binary(BinaryOp.OR, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(UnaryOp.SNEZ, instr.dst, instr.dst))
            else:
                self.seq.append(Riscv.Binary(instr.op, instr.dst, instr.lhs, instr.rhs))

        def visitAssign(self, instr: Assign) -> None:
            if instr.src is not None:
                self.seq.append(Riscv.Move(instr.dst, instr.src))

        def visitCondBranch(self, instr: CondBranch) -> None:
            self.seq.append(Riscv.Branch(instr.cond, instr.label))
        
        def visitBranch(self, instr: Branch) -> None:
            self.seq.append(Riscv.Jump(instr.target))

        # in step9, you need to think about how to pass the parameters and how to store and restore callerSave regs
        def visitParam(self, instr: Param) -> None:
            self.passed_params.append(instr.src)
        
        def visitCall(self, instr: Call) -> None:
            self.seq.append(Riscv.Call(instr.dst, self.passed_params, instr.label))
            self.passed_params.clear()

        # in step11, you need to think about how to store the array 
        def visitAlloc(self, instr: Alloc) -> None:
            self.seq.append(Riscv.GetArrayAddr(instr.dst, self.array_offset))
            self.array_offset += instr.size
"""
RiscvAsmEmitter: an SubroutineEmitter for RiscV
"""

class RiscvSubroutineEmitter(SubroutineEmitter):
    def __init__(self, emitter: RiscvAsmEmitter, info: SubroutineInfo) -> None:
        super().__init__(emitter, info)
        
        self.array_offset = info.array_offset
        # + 4 is for the RA reg 
        self.nextLocalOffset = 4 * len(Riscv.CalleeSaved) + 4 + self.array_offset
        
        # the buf which stored all the NativeInstrs in this function
        self.buf: list[NativeInstr] = []

        # from temp to int
        # record where a temp is stored in the stack
        self.offsets = {}

        self.printer.printLabel(info.funcLabel)

        # in step9, step11 you can compute the offset of local array and parameters here
        self.param_num = info.funcLabel.param_num

    def emitComment(self, comment: str) -> None:
        # you can add some log here to help you debug
        # print("riscv comment: ", comment)
        pass
    
    # store some temp to stack
    # usually happen when reaching the end of a basicblock
    # in step9, you need to think about the fuction parameters here
    def emitStoreToStack(self, src: Reg) -> None:
        if src.temp.index not in self.offsets:
            self.offsets[src.temp.index] = self.nextLocalOffset
            self.nextLocalOffset += 4
        self.buf.append(
            Riscv.NativeStoreWord(src, Riscv.SP, self.offsets[src.temp.index])
        )

    # load some temp from stack
    # usually happen when using a temp which is stored to stack before
    # in step9, you need to think about the fuction parameters here
    def emitLoadFromStack(self, dst: Reg, src: Temp):
        if src.index < self.param_num :
            if src.index < 8:
                self.buf.append(Riscv.NativeMove(eval(f"Riscv.A{src.index}"), dst))
            else:
                self.buf.append(Riscv.NativeFPLoadWord(dst, Riscv.SP, 4 * (src.index - 8)))
        elif src.index not in self.offsets:
            raise IllegalArgumentException()
        else:
            self.buf.append(Riscv.NativeLoadWord(dst, Riscv.SP, self.offsets[src.index]))

    # add a NativeInstr to buf
    # when calling the fuction emitEnd, all the instr in buf will be transformed to RiscV code
    def emitNative(self, instr: NativeInstr):
        self.buf.append(instr)

    def emitLabel(self, label: Label):
        self.buf.append(Riscv.RiscvLabel(label).toNative([], []))

    
    def emitEnd(self):
        self.printer.printComment("start of prologue")
        self.printer.printInstr(Riscv.SPAdd(-self.nextLocalOffset))

        # in step9, you need to think about how to store RA here
        # you can get some ideas from how to save CalleeSaved regs
        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeStoreWord(Riscv.CalleeSaved[i], Riscv.SP, 4 * i + self.array_offset)
                )
        self.printer.printInstr(Riscv.NativeStoreWord(Riscv.RA, Riscv.SP, 4 * len(Riscv.CalleeSaved) + self.array_offset))
        self.printer.printComment("end of prologue")
        self.printer.println("")

        self.printer.printComment("start of body")

        # in step9, you need to think about how to pass the parameters here
        # you can use the stack or regs

        # using asmcodeprinter to output the RiscV code
        for instr in self.buf:
            if isinstance(instr, Riscv.NativeFPLoadWord):
                instr.offset += self.nextLocalOffset
            self.printer.printInstr(instr)

        self.printer.printComment("end of body")
        self.printer.println("")

        self.printer.printLabel(
            Label(LabelKind.TEMP, self.info.funcLabel.name + Riscv.EPILOGUE_SUFFIX)
        )
        self.printer.printComment("start of epilogue")

        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeLoadWord(Riscv.CalleeSaved[i], Riscv.SP, 4 * i + self.array_offset)
                )
        self.printer.printInstr(Riscv.NativeLoadWord(Riscv.RA, Riscv.SP, 4 * len(Riscv.CalleeSaved) + self.array_offset))
        self.printer.printInstr(Riscv.SPAdd(self.nextLocalOffset))
        self.printer.printComment("end of epilogue")
        self.printer.println("")

        self.printer.printInstr(Riscv.NativeReturn())
        self.printer.println("")
