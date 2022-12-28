from .label import Label, LabelKind


class FuncLabel(Label):
    def __init__(self, name: str, param_num: int) -> None:
        super().__init__(LabelKind.FUNC, name)
        self.func = name
        self.param_num = param_num

    def __str__(self) -> str:
        return "FUNCTION<%s>" % self.func


MAIN_LABEL = FuncLabel("main", 0)
