import gsn.gsn as gsn
import nn.ImgNet as nn
import kb as kb

tactics = {
    "11": [],
    "01": ['self.rover.MODE="charge"', "self.isUtilityViolatedBattery = False"],
    "10": ['self.rover.MODE="manual"', "self.isUtilityViolatedCV = False"],
    "00": [
        'self.rover.MODE="charge"',
        "self.isUtilityViolatedBattery = False",
        'self.rover.MODE="manual"',
        "self.isUtilityViolatedCV = False",
    ],
}


utility_map = {
    "M0-S1.1.1": lambda uf, dp: uf.func(dp[0], dp[1]),
    "M0-S1.2.1": lambda uf, dp: uf.func(dp, False),
}


class SAS:
    def __init__(self, kbs, rover):
        self.knowledge_base = kbs
        self.rover = rover
        self.planFuncs = []
        self.utility_violation_pattern = "00"
        self.datastream = []  # array of tuples recieved from rover
        self.debug = True

    def _monitor(self):
        if self.rover.MODE == "drive":  # only monitor obs when driving
            self.rover.check_obstacle_in_radar()
        self.rover.check_rover_reaches_waypoint() 

        # 'topic' requests for monitored params
        self.datastream.append((self.rover.get_battery_data()))
        self.datastream.append(self.rover.get_cv_data())

        if self.debug:
            self.rover.print_debug_logs()

    def _analyze(self):
        self.utility_violation_pattern = ""
        for uf, dp in zip(self.knowledge_base.gsn.utility_funcs, self.datastream):
            self.utility_violation_pattern += str(int(utility_map[uf.id](uf, dp)))

    def _plan(self):
        self.planFuncs = tactics[self.utility_violation_pattern]
        if self.rover.shouldUpdateDirection:
            self.planFuncs.append("self.rover.update_direction_to_waypoint()")
            self.planFuncs.append("self.rover.shouldUpdateDirection = False")

    def _execute(self):
        for func in self.planFuncs:
            exec(func)
        self.rover.update()
        # clean up
        self.planFuncs = []
        self.datastream = []
        self.utility_violation_pattern = "00"

    def _mape_loop(self):
        self._monitor()
        self._analyze()
        self._plan()
        self._execute()
