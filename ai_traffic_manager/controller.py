# controller.py
import time
import pandas as pd

class FourWayController:
    """Adaptive 4-lane traffic controller with emissions logging."""

    def __init__(self, min_green=5, max_green=20, service_rate=2, emission_rate=0.06):
        self.lanes = ['N', 'E', 'S', 'W']
        self.queue = {l: 0 for l in self.lanes}
        self.state = {l: "RED" for l in self.lanes}
        self.current_green = 'N'
        self.green_timer = 0
        self.min_green = min_green
        self.max_green = max_green
        self.service_rate = service_rate
        self.emission_rate = emission_rate
        self.history = []
        self.set_green(self.current_green, self.min_green)

    def set_green(self, lane, green_time):
        for l in self.lanes:
            self.state[l] = "RED"
        self.state[lane] = "GREEN"
        self.current_green = lane
        self.green_timer = int(green_time)

    def step(self, counts, ambulance_lane=None, dt=1.0):
        # Add arrivals
        for l in self.lanes:
            self.queue[l] += int(counts.get(l, 0))

        mode = "NORMAL"
        if ambulance_lane:
            # immediate override
            self.set_green(ambulance_lane, self.max_green)
            mode = "EMERGENCY"
        else:
            if self.green_timer > 0:
                self.green_timer -= dt
            else:
                # pick lane with most queue
                best = max(self.lanes, key=lambda x: self.queue[x])
                q = self.queue[best]
                scale = min(q / 10.0, 1.0)
                green_time = int(self.min_green + scale * (self.max_green - self.min_green))
                self.set_green(best, green_time)

        served = min(self.queue[self.current_green], int(self.service_rate * dt))
        self.queue[self.current_green] -= served

        total_waiting = sum(self.queue.values())
        emissions = total_waiting * self.emission_rate * dt

        record = {
            "timestamp": time.time(),
            "mode": mode,
            "green_lane": self.current_green,
            "served": served,
            "queue_N": self.queue['N'],
            "queue_E": self.queue['E'],
            "queue_S": self.queue['S'],
            "queue_W": self.queue['W'],
            "total_waiting": total_waiting,
            "emissions_g": emissions
        }
        self.history.append(record)
        return record

    def export_log(self, filename="traffic_log.csv"):
        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)
        return filename


class FixedController:
    """Fixed-time cyclic controller for baseline."""

    def __init__(self, green_time=10, service_rate=2, emission_rate=0.06):
        self.lanes = ['N', 'E', 'S', 'W']
        self.queue = {l: 0 for l in self.lanes}
        self.state = {l: "RED" for l in self.lanes}
        self.current_index = 0
        self.timer = 0
        self.green_time = green_time
        self.service_rate = service_rate
        self.emission_rate = emission_rate
        self.history = []
        self.set_green(self.lanes[self.current_index])

    def set_green(self, lane):
        for l in self.lanes:
            self.state[l] = "RED"
        self.state[lane] = "GREEN"
        self.current_green = lane
        self.timer = self.green_time

    def step(self, counts, dt=1.0):
        for l in self.lanes:
            self.queue[l] += int(counts.get(l, 0))

        if self.timer > 0:
            self.timer -= dt
        else:
            self.current_index = (self.current_index + 1) % 4
            self.set_green(self.lanes[self.current_index])

        served = min(self.queue[self.current_green], int(self.service_rate * dt))
        self.queue[self.current_green] -= served

        total_waiting = sum(self.queue.values())
        emissions = total_waiting * self.emission_rate * dt

        record = {
            "timestamp": time.time(),
            "mode": "FIXED",
            "green_lane": self.current_green,
            "served": served,
            "queue_N": self.queue['N'],
            "queue_E": self.queue['E'],
            "queue_S": self.queue['S'],
            "queue_W": self.queue['W'],
            "total_waiting": total_waiting,
            "emissions_g": emissions
        }
        self.history.append(record)
        return record
