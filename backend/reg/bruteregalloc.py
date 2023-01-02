import random

from backend.dataflow.basicblock import BasicBlock, BlockKind
from backend.dataflow.cfg import CFG
from backend.dataflow.loc import Loc
from backend.reg.regalloc import RegAlloc
from backend.riscv.riscvasmemitter import RiscvAsmEmitter
from backend.subroutineemitter import SubroutineEmitter
from backend.subroutineinfo import SubroutineInfo
from utils.riscv import Riscv
from utils.tac.holeinstr import HoleInstr
from utils.tac.reg import Reg
from utils.tac.temp import Temp

"""
BruteRegAlloc: one kind of RegAlloc

bindings: map from temp.index to Reg

we don't need to take care of GlobalTemp here
because we can remove all the GlobalTemp in selectInstr process

1. accept：根据每个函数的 CFG 进行寄存器分配，寄存器分配结束后生成相应汇编代码
2. bind：将一个 Temp 与寄存器绑定
3. unbind：将一个 Temp 与相应寄存器解绑定
4. localAlloc：根据数据流对一个 BasicBlock 内的指令进行寄存器分配
5. allocForLoc：每一条指令进行寄存器分配
6. allocRegFor：根据数据流决定为当前 Temp 分配哪一个寄存器
"""

class BruteRegAlloc(RegAlloc):
    def __init__(self, emitter: RiscvAsmEmitter) -> None:
        super().__init__(emitter)
        self.bindings = {}
        for reg in emitter.allocatableRegs:
            reg.used = False

    def accept(self, graph: CFG, info: SubroutineInfo, numArgs: int) -> None:
        subEmitter = self.emitter.emitSubroutine(info)
        for i in range(numArgs):
            self.bind(Temp(i), eval(f"Riscv.A{i}"))
        for bb in graph.iterator():
            # you need to think more here
            # maybe we don't need to alloc regs for all the basic blocks
            if bb.label is not None:
                subEmitter.emitLabel(bb.label)
            if graph.isAccessible(bb.id):
                self.localAlloc(bb, subEmitter)
        subEmitter.emitEnd()

    def bind(self, temp: Temp, reg: Reg):
        reg.used = True
        self.bindings[temp.index] = reg
        reg.occupied = True
        reg.temp = temp

    def unbind(self, temp: Temp):
        if temp.index in self.bindings:
            self.bindings[temp.index].occupied = False
            self.bindings.pop(temp.index)

    def localAlloc(self, bb: BasicBlock, subEmitter: SubroutineEmitter):
        self.bindings.clear()
        for reg in self.emitter.allocatableRegs:
            reg.occupied = False

        # in step9, you may need to think about how to store callersave regs here
        for loc in bb.allSeq():
            subEmitter.emitComment(str(loc.instr))

            self.allocForLoc(loc, subEmitter)

        for tempindex in bb.liveOut:
            if tempindex in self.bindings:
                subEmitter.emitStoreToStack(self.bindings.get(tempindex))

        if (not bb.isEmpty()) and (bb.kind is not BlockKind.CONTINUOUS):
            self.allocForLoc(bb.locs[len(bb.locs) - 1], subEmitter)

    def allocForLoc(self, loc: Loc, subEmitter: SubroutineEmitter):
        instr = loc.instr
        srcRegs: list[Reg] = []
        dstRegs: list[Reg] = []

        paramRegs = []
        if isinstance(instr, Riscv.Call):
            paramRegs = [Riscv.A0, Riscv.A1, Riscv.A2, Riscv.A3, Riscv.A4, Riscv.A5, Riscv.A6, Riscv.A7]
        for i in range(len(instr.srcs)):
            temp = instr.srcs[i]
            if isinstance(temp, Reg):
                srcRegs.append(temp)
            else:
                srcRegs.append(self.allocRegFor(temp, True, loc.liveIn, subEmitter, paramRegs))

        usedRegs = []
        if isinstance(instr, Riscv.Call):
            for reg in self.emitter.callerSaveRegs:
                if reg.isUsed():
                    usedRegs.append(reg)

        for i in range(len(instr.dsts)):
            temp = instr.dsts[i]
            if isinstance(temp, Reg):
                dstRegs.append(temp)
            else:
                dstRegs.append(self.allocRegFor(temp, False, loc.liveIn, subEmitter, usedRegs))

        if isinstance(instr, Riscv.Call):
            subEmitter.emitNative(Riscv.SPAdd(-4 * (len(usedRegs) + (max(len(srcRegs) - 8, 0)))))
            offset = 0
            if len(srcRegs) > 8:
                for i in range(8, len(srcRegs)):
                    subEmitter.emitNative(
                        Riscv.NativeStoreWord(srcRegs[i], Riscv.SP, offset)
                    )
                    offset += 4
            param_offset = offset
            for reg in usedRegs:
                subEmitter.emitNative(
                    Riscv.NativeStoreWord(reg, Riscv.SP, offset)
                )
                offset += 4
            for i in range(min(len(srcRegs), 8)):
                subEmitter.emitNative(Riscv.NativeMove(srcRegs[i], eval(f"Riscv.A{i}")))
            subEmitter.emitNative(instr.toNative(dstRegs, srcRegs))
            subEmitter.emitNative(Riscv.NativeMove(Riscv.A0, dstRegs[0]))
            for reg in usedRegs:
                subEmitter.emitNative(
                    Riscv.NativeLoadWord(reg, Riscv.SP, param_offset)
                )
                param_offset += 4
            subEmitter.emitNative(
                Riscv.SPAdd(4 * (len(usedRegs) + (max(len(srcRegs) - 8, 0))))
            )
        else:
            subEmitter.emitNative(instr.toNative(dstRegs, srcRegs))
    

    def allocRegFor(
        self, temp: Temp, isRead: bool, live: set[int], subEmitter: SubroutineEmitter, usedRegs: list[Reg] = []
    ):
        if temp.index in self.bindings:
            return self.bindings[temp.index]

        allocatableRegs = []
        for reg in self.emitter.allocatableRegs:
            if reg not in usedRegs:
                allocatableRegs.append(reg)
        for reg in allocatableRegs:
            if (not reg.occupied) or (not reg.temp.index in live):
                subEmitter.emitComment(
                    "  allocate {} to {}  (read: {}):".format(
                        str(temp), str(reg), str(isRead)
                    )
                )
                if isRead:
                    subEmitter.emitLoadFromStack(reg, temp)
                if reg.occupied:
                    self.unbind(reg.temp)
                self.bind(temp, reg)
                return reg

        reg = allocatableRegs[random.randint(0, len(allocatableRegs) - 1)]
        subEmitter.emitStoreToStack(reg)
        subEmitter.emitComment("  spill {} ({})".format(str(reg), str(reg.temp)))
        self.unbind(reg.temp)
        self.bind(temp, reg)
        subEmitter.emitComment(
            "  allocate {} to {} (read: {})".format(str(temp), str(reg), str(isRead))
        )
        if isRead:
            subEmitter.emitLoadFromStack(reg, temp)
        return reg
