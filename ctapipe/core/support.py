class Singleton(type):
    """" metaclass for singleton pattern """
    instance = None
    def __call__(cls, *args, **kw):
        if not cls.instance:
             cls.instance = super(Singleton, cls).__call__(*args, **kw)
        return cls.instance

