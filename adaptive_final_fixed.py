"""
AI Adaptive Traffic Management System
======================================
A hackathon-ready simulation demonstrating adaptive traffic signal control
with emergency vehicle prioritization and comprehensive metrics tracking.

Requirements:
    pip install pygame opencv-python numpy pandas

Usage:
    python adaptive_final_fixed.py
"""

import pygame
import numpy as np
import cv2
import pandas as pd
from collections import deque
import time
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

# Display settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
FPS = 30
SIMULATION_DURATION = 60  # seconds

# Road layout
ROAD_WIDTH = 120
LANE_WIDTH = 30
CENTER_SIZE = 140

# Vehicle settings
VEHICLE_LENGTH = 35
VEHICLE_WIDTH = 20
VEHICLE_SPEED = 2.5
VEHICLE_SPACING = 45
SPAWN_INTERVAL = 45  # frames between spawns

# Traffic control
MIN_GREEN_TIME = 4.0
MAX_GREEN_TIME = 10.0
YELLOW_TIME = 1.5
CLEAR_TIME = 0.8
STARVATION_THRESHOLD = 15.0  # seconds

# Ambulance settings
AMBULANCE_SPAWN_CHANCE = 0.015  # 1.5% chance per spawn

# Metrics
IDLE_CO2_RATE = 0.06  # grams per second per vehicle

# Colors
COLOR_BG = (25, 25, 30)
COLOR_ROAD = (45, 45, 50)
COLOR_STRIPE = (200, 200, 80)
COLOR_ZEBRA = (220, 220, 220)
COLOR_TEXT = (220, 220, 220)
COLOR_RED = (255, 60, 60)
COLOR_YELLOW = (255, 200, 60)
COLOR_GREEN = (60, 255, 120)

LANE_COLORS = {
    'N': (255, 80, 80),   # Red
    'E': (80, 150, 255),  # Blue
    'S': (80, 255, 120),  # Green
    'W': (255, 200, 80)   # Yellow
}

# ============================================================================
# VEHICLE CLASS
# ============================================================================

class Vehicle:
    def __init__(self, lane, position, is_ambulance=False):
        self.lane = lane
        self.position = position  # Distance from center (starts far, moves to 0)
        self.speed = VEHICLE_SPEED
        self.is_ambulance = is_ambulance
        self.pulse_phase = 0
        
    def update(self, can_proceed, vehicle_ahead):
        """Update vehicle position with collision avoidance"""
        target_speed = VEHICLE_SPEED
        
        # Stop for red light when approaching intersection
        if not can_proceed and self.position > 80:
            target_speed = 0
            
        # Maintain spacing from vehicle ahead
        if vehicle_ahead:
            distance = self.position - vehicle_ahead.position
            if distance < VEHICLE_SPACING:
                target_speed = min(target_speed, vehicle_ahead.speed * 0.8)
                
        # Smooth acceleration/deceleration
        self.speed += (target_speed - self.speed) * 0.2
        self.position -= self.speed  # Move TOWARD center (decrease position)
        
        # Pulse animation for ambulance
        if self.is_ambulance:
            self.pulse_phase = (self.pulse_phase + 0.15) % (2 * np.pi)
            
        return self.position < -100  # Remove if passed through intersection

# ============================================================================
# ADAPTIVE TRAFFIC CONTROLLER
# ============================================================================

class AdaptiveController:
    def __init__(self):
        self.lanes = ['N', 'E', 'S', 'W']
        self.current_lane = 0
        self.state = 'GREEN'
        self.state_timer = 0
        self.green_duration = MIN_GREEN_TIME
        self.last_green_time = {lane: 0 for lane in self.lanes}
        self.queue_history = {lane: deque(maxlen=30) for lane in self.lanes}
        self.emergency_active = False
        self.emergency_lane = None
        
    def trigger_emergency(self, lane):
        """Immediately switch to emergency mode for ambulance"""
        if not self.emergency_active:
            self.emergency_active = True
            self.emergency_lane = lane
            self.current_lane = self.lanes.index(lane)
            self.state = 'GREEN'
            self.state_timer = 0
            self.green_duration = 8.0
            return True
        return False
        
    def update(self, dt, queues, ambulance_lane):
        """Adaptive signal control logic"""
        self.state_timer += dt
        
        # Update queue history for smoothing
        for lane in self.lanes:
            self.queue_history[lane].append(queues[lane])
            
        # Check if emergency is still active
        if self.emergency_active and ambulance_lane != self.emergency_lane:
            self.emergency_active = False
            self.emergency_lane = None
            
        # State machine
        if self.state == 'GREEN':
            if self.state_timer >= self.green_duration:
                self.state = 'YELLOW'
                self.state_timer = 0
                self.last_green_time[self.lanes[self.current_lane]] = time.time()
                
        elif self.state == 'YELLOW':
            if self.state_timer >= YELLOW_TIME:
                self.state = 'CLEAR'
                self.state_timer = 0
                
        elif self.state == 'CLEAR':
            if self.state_timer >= CLEAR_TIME:
                # Select next lane adaptively
                if not self.emergency_active:
                    self.current_lane = self._select_next_lane(queues)
                    self.green_duration = self._calculate_green_duration(queues)
                self.state = 'GREEN'
                self.state_timer = 0
                
        return self.lanes[self.current_lane]
    
    def _select_next_lane(self, queues):
        """Select next lane based on queue length and starvation prevention"""
        current_time = time.time()
        scores = []
        
        for i, lane in enumerate(self.lanes):
            # Smoothed queue length
            avg_queue = np.mean(self.queue_history[lane]) if self.queue_history[lane] else 0
            
            # Time since last green
            wait_time = current_time - self.last_green_time[lane]
            
            # Score: prioritize long queues and starving lanes
            starvation_bonus = 2.0 if wait_time > STARVATION_THRESHOLD else 1.0
            score = avg_queue * starvation_bonus + wait_time * 0.1
            scores.append(score)
            
        return int(np.argmax(scores))
    
    def _calculate_green_duration(self, queues):
        """Calculate optimal green duration based on queue density"""
        lane = self.lanes[self.current_lane]
        avg_queue = np.mean(self.queue_history[lane]) if self.queue_history[lane] else 0
        
        # Linear interpolation between min and max
        if avg_queue <= 2:
            return MIN_GREEN_TIME
        elif avg_queue >= 8:
            return MAX_GREEN_TIME
        else:
            ratio = (avg_queue - 2) / 6
            return MIN_GREEN_TIME + ratio * (MAX_GREEN_TIME - MIN_GREEN_TIME)
    
    def can_proceed(self, lane):
        """Check if vehicles in a lane can proceed"""
        return self.state == 'GREEN' and self.lanes[self.current_lane] == lane

# ============================================================================
# RENDERING FUNCTIONS
# ============================================================================

def draw_intersection(screen):
    """Draw the intersection background"""
    screen.fill(COLOR_BG)
    
    center_x = WINDOW_WIDTH // 2
    center_y = WINDOW_HEIGHT // 2
    
    # Draw roads (N-S and E-W)
    pygame.draw.rect(screen, COLOR_ROAD, 
                     (center_x - ROAD_WIDTH//2, 0, ROAD_WIDTH, WINDOW_HEIGHT))
    pygame.draw.rect(screen, COLOR_ROAD, 
                     (0, center_y - ROAD_WIDTH//2, WINDOW_WIDTH, ROAD_WIDTH))
    
    # Center intersection box
    pygame.draw.rect(screen, COLOR_ROAD,
                     (center_x - CENTER_SIZE//2, center_y - CENTER_SIZE//2, 
                      CENTER_SIZE, CENTER_SIZE))
    
    # Lane dividers (dashed lines)
    dash_length = 20
    gap_length = 15
    
    # Vertical divider
    y = 0
    while y < WINDOW_HEIGHT:
        if not (center_y - CENTER_SIZE//2 < y < center_y + CENTER_SIZE//2):
            pygame.draw.line(screen, COLOR_STRIPE,
                           (center_x, y), (center_x, y + dash_length), 2)
        y += dash_length + gap_length
        
    # Horizontal divider
    x = 0
    while x < WINDOW_WIDTH:
        if not (center_x - CENTER_SIZE//2 < x < center_x + CENTER_SIZE//2):
            pygame.draw.line(screen, COLOR_STRIPE,
                           (x, center_y), (x + dash_length, center_y), 2)
        x += dash_length + gap_length
    
    # Zebra crossings (FIXED positioning)
    stripe_width = 10
    stripe_gap = 4
    crossing_length = 80
    
    # North crossing (above intersection)
    for i in range(6):
        offset = i * (stripe_width + stripe_gap)
        pygame.draw.rect(screen, COLOR_ZEBRA,
                        (center_x - crossing_length//2 + offset, 
                         center_y - CENTER_SIZE//2 - 40,
                         stripe_width, 30))
    
    # South crossing (below intersection)
    for i in range(6):
        offset = i * (stripe_width + stripe_gap)
        pygame.draw.rect(screen, COLOR_ZEBRA,
                        (center_x - crossing_length//2 + offset,
                         center_y + CENTER_SIZE//2 + 10,
                         stripe_width, 30))
    
    # East crossing (right of intersection)
    for i in range(6):
        offset = i * (stripe_width + stripe_gap)
        pygame.draw.rect(screen, COLOR_ZEBRA,
                        (center_x + CENTER_SIZE//2 + 10,
                         center_y - crossing_length//2 + offset,
                         30, stripe_width))
    
    # West crossing (left of intersection)
    for i in range(6):
        offset = i * (stripe_width + stripe_gap)
        pygame.draw.rect(screen, COLOR_ZEBRA,
                        (center_x - CENTER_SIZE//2 - 40,
                         center_y - crossing_length//2 + offset,
                         30, stripe_width))

def get_vehicle_screen_pos(lane, position):
    """Convert lane position to screen coordinates"""
    center_x = WINDOW_WIDTH // 2
    center_y = WINDOW_HEIGHT // 2
    
    # Position is distance from center (positive = far from center)
    if lane == 'N':
        # Coming from top, moving down toward center
        return (center_x - LANE_WIDTH//2 + 7, center_y - position)
    elif lane == 'S':
        # Coming from bottom, moving up toward center
        return (center_x + LANE_WIDTH//2 - 7, center_y + position)
    elif lane == 'E':
        # Coming from right, moving left toward center
        return (center_x + position, center_y + LANE_WIDTH//2 - 7)
    else:  # W
        # Coming from left, moving right toward center
        return (center_x - position, center_y - LANE_WIDTH//2 + 7)

def draw_vehicle(screen, vehicle):
    """Draw a single vehicle"""
    x, y = get_vehicle_screen_pos(vehicle.lane, vehicle.position)
    color = LANE_COLORS[vehicle.lane]
    
    # Determine orientation
    if vehicle.lane in ['N', 'S']:
        rect = pygame.Rect(x - VEHICLE_WIDTH//2, y - VEHICLE_LENGTH//2,
                          VEHICLE_WIDTH, VEHICLE_LENGTH)
    else:
        rect = pygame.Rect(x - VEHICLE_LENGTH//2, y - VEHICLE_WIDTH//2,
                          VEHICLE_LENGTH, VEHICLE_WIDTH)
    
    if vehicle.is_ambulance:
        # White ambulance with pulsing red border
        pygame.draw.rect(screen, (255, 255, 255), rect, border_radius=3)
        pulse = int(100 + 155 * (np.sin(vehicle.pulse_phase) + 1) / 2)
        pygame.draw.rect(screen, (pulse, 0, 0), rect, 3, border_radius=3)
        
        # Add red cross symbol
        cross_size = 8
        cross_x, cross_y = rect.center
        pygame.draw.rect(screen, (255, 0, 0), 
                        (cross_x - 2, cross_y - cross_size//2, 4, cross_size))
        pygame.draw.rect(screen, (255, 0, 0), 
                        (cross_x - cross_size//2, cross_y - 2, cross_size, 4))
    else:
        pygame.draw.rect(screen, color, rect, border_radius=3)
        # Darker outline
        darker = tuple(max(0, c - 40) for c in color)
        pygame.draw.rect(screen, darker, rect, 2, border_radius=3)

def draw_traffic_light(screen, lane, state, is_active):
    """Draw traffic signal for a lane"""
    center_x = WINDOW_WIDTH // 2
    center_y = WINDOW_HEIGHT // 2
    offset = CENTER_SIZE // 2 + 50
    
    positions = {
        'N': (center_x + 30, center_y - offset),
        'S': (center_x - 30, center_y + offset),
        'E': (center_x + offset, center_y - 30),
        'W': (center_x - offset, center_y + 30)
    }
    
    x, y = positions[lane]
    
    # Signal box
    pygame.draw.rect(screen, (30, 30, 35), (x - 12, y - 35, 24, 75), border_radius=4)
    
    # Lights
    red_color = COLOR_RED if (not is_active or state in ['YELLOW', 'CLEAR']) else (80, 20, 20)
    yellow_color = COLOR_YELLOW if (is_active and state == 'YELLOW') else (80, 60, 20)
    green_color = COLOR_GREEN if (is_active and state == 'GREEN') else (20, 80, 30)
    
    pygame.draw.circle(screen, red_color, (x, y - 20), 8)
    pygame.draw.circle(screen, yellow_color, (x, y), 8)
    pygame.draw.circle(screen, green_color, (x, y + 20), 8)
    
    # Glow effect for active light
    if is_active:
        if state == 'GREEN':
            pygame.draw.circle(screen, green_color, (x, y + 20), 12, 2)
        elif state == 'YELLOW':
            pygame.draw.circle(screen, yellow_color, (x, y), 12, 2)

def draw_hud(screen, controller, queues, metrics, font, emergency_active):
    """Draw heads-up display with metrics"""
    y_offset = 20
    
    # Emergency indicator
    if emergency_active:
        emergency_text = "🚨 EMERGENCY MODE - AMBULANCE PRIORITY 🚨"
        text_surf = font.render(emergency_text, True, (255, 100, 100))
        screen.blit(text_surf, (20, y_offset))
        y_offset += 40
    
    # State indicator
    state_colors = {'GREEN': COLOR_GREEN, 'YELLOW': COLOR_YELLOW, 'CLEAR': COLOR_RED}
    state_text = f"STATE: {controller.state}"
    text_surf = font.render(state_text, True, state_colors[controller.state])
    screen.blit(text_surf, (20, y_offset))
    y_offset += 35
    
    # Current green lane and timer
    green_lane = controller.lanes[controller.current_lane]
    remaining = controller.green_duration - controller.state_timer
    timer_text = f"Green: {green_lane}  |  Time: {remaining:.1f}s"
    text_surf = font.render(timer_text, True, COLOR_TEXT)
    screen.blit(text_surf, (20, y_offset))
    y_offset += 35
    
    # Queue counts
    queue_text = f"Queues → N:{queues['N']} E:{queues['E']} S:{queues['S']} W:{queues['W']}"
    text_surf = font.render(queue_text, True, COLOR_TEXT)
    screen.blit(text_surf, (20, y_offset))
    y_offset += 35
    
    # Total waiting
    total_waiting = sum(queues.values())
    wait_text = f"Total Waiting: {total_waiting}"
    text_surf = font.render(wait_text, True, COLOR_TEXT)
    screen.blit(text_surf, (20, y_offset))
    y_offset += 35
    
    # Served vehicles
    served_text = f"Vehicles Served: {metrics['served']}"
    text_surf = font.render(served_text, True, COLOR_TEXT)
    screen.blit(text_surf, (20, y_offset))
    y_offset += 35
    
    # Emissions
    emissions_text = f"Avg Idle Emissions: {metrics['avg_emissions']:.2f} g/s"
    text_surf = font.render(emissions_text, True, COLOR_TEXT)
    screen.blit(text_surf, (20, y_offset))

# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_simulation():
    """Execute the complete traffic simulation"""
    
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("AI Adaptive Traffic Management System")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('adaptive_final.mp4', fourcc, FPS, 
                                   (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Initialize simulation state
    controller = AdaptiveController()
    vehicles = {lane: [] for lane in ['N', 'E', 'S', 'W']}
    spawn_counters = {lane: 0 for lane in ['N', 'E', 'S', 'W']}
    
    metrics = {
        'served': 0,
        'total_emissions': 0,
        'avg_emissions': 0,
        'frames': 0
    }
    
    log_data = []
    
    # Simulation loop
    running = True
    start_time = time.time()
    
    while running:
        dt = 1 / FPS
        elapsed = time.time() - start_time
        
        if elapsed > SIMULATION_DURATION:
            running = False
            
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Spawn vehicles
        for lane in ['N', 'E', 'S', 'W']:
            spawn_counters[lane] += 1
            if spawn_counters[lane] >= SPAWN_INTERVAL:
                spawn_counters[lane] = 0
                
                # Random ambulance spawn
                is_ambulance = random.random() < AMBULANCE_SPAWN_CHANCE
                
                # Spawn far from center (vehicles move toward center)
                new_vehicle = Vehicle(lane, 400, is_ambulance)
                vehicles[lane].append(new_vehicle)
                
                # Trigger emergency if ambulance spawned
                if is_ambulance:
                    controller.trigger_emergency(lane)
        
        # Update controller
        queues = {lane: len(vehicles[lane]) for lane in ['N', 'E', 'S', 'W']}
        ambulance_lane = None
        for lane, vehs in vehicles.items():
            if any(v.is_ambulance for v in vehs):
                ambulance_lane = lane
                break
                
        green_lane = controller.update(dt, queues, ambulance_lane)
        
        # Update vehicles
        for lane in ['N', 'E', 'S', 'W']:
            can_proceed = controller.can_proceed(lane)
            to_remove = []
            
            for i, vehicle in enumerate(vehicles[lane]):
                vehicle_ahead = vehicles[lane][i-1] if i > 0 else None
                crossed = vehicle.update(can_proceed, vehicle_ahead)
                
                if crossed:
                    to_remove.append(i)
                    metrics['served'] += 1
                    
            # Remove crossed vehicles
            for i in reversed(to_remove):
                vehicles[lane].pop(i)
        
        # Calculate emissions
        total_waiting = sum(queues.values())
        frame_emissions = total_waiting * IDLE_CO2_RATE / FPS
        metrics['total_emissions'] += frame_emissions
        metrics['frames'] += 1
        metrics['avg_emissions'] = metrics['total_emissions'] / metrics['frames']
        
        # Log data
        log_data.append({
            'time_s': elapsed,
            'green': green_lane,
            'state': controller.state,
            'q_N': queues['N'],
            'q_E': queues['E'],
            'q_S': queues['S'],
            'q_W': queues['W'],
            'total_waiting': total_waiting,
            'served_total': metrics['served'],
            'emissions_g_frame': frame_emissions,
            'emergency_active': controller.emergency_active
        })
        
        # Render
        draw_intersection(screen)
        
        # Draw traffic lights
        for lane in ['N', 'E', 'S', 'W']:
            is_active = (lane == green_lane)
            draw_traffic_light(screen, lane, controller.state, is_active)
        
        # Draw vehicles
        for lane in ['N', 'E', 'S', 'W']:
            for vehicle in vehicles[lane]:
                draw_vehicle(screen, vehicle)
        
        # Draw HUD
        draw_hud(screen, controller, queues, metrics, font, controller.emergency_active)
        
        # Capture frame for video
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    # Cleanup
    video_writer.release()
    pygame.quit()
    
    # Save CSV
    df = pd.DataFrame(log_data)
    df.to_csv('adaptive_final.csv', index=False)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Total vehicles served: {metrics['served']}")
    print(f"Average emissions: {metrics['avg_emissions']:.2f} g/s")
    print(f"\nOutput files generated:")
    print("  • adaptive_final.mp4 (video)")
    print("  • adaptive_final.csv (metrics log)")
    print("="*60)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI ADAPTIVE TRAFFIC MANAGEMENT SYSTEM")
    print("="*60)
    print("Starting simulation...")
    print(f"Duration: {SIMULATION_DURATION} seconds")
    print(f"Resolution: {WINDOW_WIDTH}x{WINDOW_HEIGHT} @ {FPS} FPS")
    print("="*60 + "\n")
    
    run_simulation()