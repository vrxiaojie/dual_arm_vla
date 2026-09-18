from enum import Enum, auto


class TaskState(Enum):
    INIT = auto()
    PRE_GRASP = auto()
    APPROACH = auto()
    GRASP = auto()
    CHECK_GRASP = auto()
    LIFT = auto()
    TRANSPORT = auto()
    PLACE = auto()
    RELEASE = auto()
    RETREAT = auto()
    SUCCESS = auto()
    FAILURE_STOP = auto()


ALLOWED_TRANSITIONS = {
    TaskState.INIT: TaskState.PRE_GRASP,
    TaskState.PRE_GRASP: TaskState.APPROACH,
    TaskState.APPROACH: TaskState.GRASP,
    TaskState.GRASP: TaskState.CHECK_GRASP,
    TaskState.CHECK_GRASP: TaskState.LIFT,
    TaskState.LIFT: TaskState.TRANSPORT,
    TaskState.TRANSPORT: TaskState.PLACE,
    TaskState.PLACE: TaskState.RELEASE,
    TaskState.RELEASE: TaskState.RETREAT,
    TaskState.RETREAT: TaskState.SUCCESS,
}


class StateMachine:
    def __init__(self):
        self.state = TaskState.INIT
        self.history = [self.state.name]

    def advance(self):
        self.state = ALLOWED_TRANSITIONS[self.state]
        self.history.append(self.state.name)

    def fail(self):
        self.state = TaskState.FAILURE_STOP
        self.history.append(self.state.name)

