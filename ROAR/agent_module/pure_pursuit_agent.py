from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from pathlib import Path
from ROAR.control_module.pure_pursuit_control import PurePursuitController
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import \
    WaypointFollowingMissionPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import \
    BehaviorPlanner
from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.kalman_filter.KalmanFilter import Kalman_Filter
import numpy as np
import time


class PurePursuitAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, target_speed=50):
        super().__init__(vehicle=vehicle, agent_settings=agent_settings)
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.pure_pursuit_controller = \
            PurePursuitController(agent=self,
                                  target_speed=target_speed,
                                  look_ahead_gain=0.1,
                                  look_ahead_distance=3)
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)

        # initiated right after mission plan
        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = SimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.pure_pursuit_controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=3)

        initial_state = np.array([8.7, 53.3, 0, np.pi/2])
        delta_T = 0.5
        self.kalman_filter = Kalman_Filter(initial_state,delta_T)
        self.previous_speed = 0
        self.previous_time = 0
        self.MAX_STEERING_ANGLE = 0.7854

    def run_step(self, sensors_data: SensorsData,
                 vehicle: Vehicle) -> VehicleControl:
        super(PurePursuitAgent, self).run_step(sensors_data=sensors_data,
                                               vehicle=vehicle)

        control = self.local_planner.run_in_series()
        current_time = time.time()
        print(self.kalman_filter.current_state)

        current_speed = self.vehicle.get_speed(self.vehicle)/3.6
        current_acceleration = (current_speed-self.previous_speed)/(current_time-self.previous_time)
        heading_angle = -self.vehicle.transform.rotation.yaw * np.pi * 2 / 360
        self.kalman_filter.update([self.vehicle.transform.location.x,self.vehicle.transform.location.y,current_speed,heading_angle],[current_acceleration, self.MAX_STEERING_ANGLE*control.steering])
        prediction_time = 4
        future_states = self.kalman_filter.predict_plot_future(prediction_time, plot=False)
        safe_states, predicted_collision_time =self.kalman_filter.find_collision_state(future_states)
        THRESHOLD_COLLISION_TIME = 1.5
        self.previous_speed = current_speed
        self.previous_time = current_time

        end = time.time()
        # print('time: '+str(1/(end - start)))

        if predicted_collision_time is not None and predicted_collision_time <= THRESHOLD_COLLISION_TIME:
            print('Braking Vehicle')
            return VehicleControl()
        else:
            return control
