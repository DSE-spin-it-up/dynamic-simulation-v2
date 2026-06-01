from statemachine import StateChart, State

class MissionManager(StateChart):
    take_off = State(initial=True)
    spinning_ground = State()
    spinning_up = State()