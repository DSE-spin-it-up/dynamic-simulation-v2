class TrajectoryPlanner():
    def __init__(self, mission_phase=0):
        self.mission_phase = mission_phase
        self.payload_target = 0.0
        self.payload_target_t = 0.0
        self.next_traj_step_t = 0.0

    def set_payload_target(self, target, target_time):
        #TODO: In the future, use the mission phase or smth to update the payload target instead of taking it as input
        self.payload_target = target
        self.payload_target_t = target_time