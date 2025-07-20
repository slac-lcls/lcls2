class TransitionId:
    # Enum-style values
    ClearReadout        = 0
    Reset               = 1
    Configure           = 2
    Unconfigure         = 3
    BeginRun            = 4
    EndRun              = 5
    BeginStep           = 6
    EndStep             = 7
    Enable              = 8
    Disable             = 9
    SlowUpdate          = 10
    L1Accept_EndOfBatch = 11
    L1Accept            = 12
    NumberOf            = 13

    # Internal name mapping
    _id_to_name = {
        0: "ClearReadout",
        1: "Reset",
        2: "Configure",
        3: "Unconfigure",
        4: "BeginRun",
        5: "EndRun",
        6: "BeginStep",
        7: "EndStep",
        8: "Enable",
        9: "Disable",
        10: "SlowUpdate",
        11: "L1Accept_EndOfBatch",
        12: "L1Accept",
        13: "NumberOf"
    }

    _name_to_id = {v: k for k, v in _id_to_name.items()}

    @classmethod
    def name(cls, transition_id):
        """
        Get the name string of a transition ID.

        Args:
            transition_id (int): Numeric transition ID.

        Returns:
            str: Human-readable transition name.
        """
        return cls._id_to_name.get(transition_id, f"Unknown({transition_id})")

    @classmethod
    def value(cls, name):
        """
        Get the transition ID from a name.

        Args:
            name (str): Transition name (e.g. 'BeginRun').

        Returns:
            int: Transition ID.
        """
        return cls._name_to_id.get(name)

    @classmethod
    def isEvent(cls, transition_id):
        """
        Return True if the ID corresponds to an event transition.

        Args:
            transition_id (int): Transition ID.

        Returns:
            bool: True if L1Accept or L1Accept_EndOfBatch.
        """
        return transition_id in (cls.L1Accept, cls.L1Accept_EndOfBatch)

    @classmethod
    def all_ids(cls):
        """Return all known transition IDs as a list of ints."""
        return list(cls._id_to_name.keys())
