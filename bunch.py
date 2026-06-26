class Bunch(dict):
    """兼容旧 FR-UNet checkpoint 反序列化所需的 bunch.Bunch 类型。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

