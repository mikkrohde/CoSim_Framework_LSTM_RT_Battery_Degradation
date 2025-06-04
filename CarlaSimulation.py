import random
import socket
import json
import carla
import time
import csv
import numpy as np
import signal
import re
from datetime import datetime


class SimulinkCarlaCoSimulation:
    """
    Matlab functions as the master
    """
    def __init__(self, map_name="Town03", carla_port=4000, simulink_port=3001, sync_mode=True, spawn_hero=True, init_vehicles=10, min_vehicles=10, max_vehicles=20, veh_upd_interval = 60, weather_preset_idx=0, soc_init=0.75, simulation_profile="A1"):
        self.SYNC_MODE = sync_mode
        self.SPAWN_HERO = spawn_hero
        self.atmPressureKPA = 101.3
        self.client = None
        self.world = None
        self.blueprints = None
        self.settings = None

        self.map_name = map_name
        self.weather_preset = weather_preset_idx
        self.simulation_profile = simulation_profile
        self.initSOC = soc_init

        self.init_vehicles = init_vehicles
        self.min_vehicles = min_vehicles
        self.max_vehicles = max_vehicles
        self.veh_upd_interval = veh_upd_interval
        
        #Vehicle data/information (must match in MATLAB)
        self.MAX_POWER = 150.0 #150kW
        self.vehicle_min_RPM = 0.0
        self.vehicle_max_RPM = 0.0
        self.vehicle_measured_RPM = 0.0
        self.weather_preset_idx = weather_preset_idx

        # Environmental settings
        self.MIN_TEMP = -5
        self.MAX_TEMP = 45
        self.CURRENT_TEMP = 15
        self.TEMP_CHANGE_RATE = 0.5
        self.WEATHER_PRESETS = [
            {"temp_range": (-10, 0), "weather": carla.WeatherParameters.WetCloudyNoon},
            {"temp_range": (5, 10), "weather": carla.WeatherParameters.CloudyNoon},
            {"temp_range": (20, 25), "weather": carla.WeatherParameters.ClearNoon},
            {"temp_range": (25, 35), "weather": carla.WeatherParameters.ClearSunset},
            {"temp_range": (35, 45), "weather": carla.WeatherParameters.MidRainyNoon},
        ]

        self.selected_preset = self.WEATHER_PRESETS[weather_preset_idx]
        self.MIN_TEMP, self.MAX_TEMP = self.selected_preset["temp_range"]
        self.CURRENT_TEMP = np.clip(
            (self.MIN_TEMP + self.MAX_TEMP) / 2,  # Start at midpoint
            self.MIN_TEMP,
            self.MAX_TEMP
        )
        
        # Time tracking for environmental updates
        self.last_temp_update = 0.0
        self.last_traffic_light_update = 0.0
        self.TEMP_UPDATE_INTERVAL = 20.0
        self.TRAFFIC_LIGHT_INTERVAL = 20.0

        # Data storage initialization
        self.last_simulink_data = {
            'simulation_time': 0.0,
            'dt_physics': 0.0,
            'battery_age_factor' : 0.0,
            'time_battery_accel': 0.0,
            'vehicle_speed' : 0.0,
            'throttle' : 0.0,
            'front_rpm' : 0.0,
            'rear_rpm' : 0.0, 
            'range' : 0.0,
            'battery_degradation' : 0.0,
            'battery_soh' : 0.0,
            'battery_temp' : 0.0,
            'battery_soc' : 0.0, 
            'battery_curr' : 0.0,
            'battery_cooling_temp' : 0.0,
        }        

        self.telemetry = {
            'throttle'  : 0.0,
            'speed'     : 0.0,
            'roll'      : 0.0,
            'yaw'       : 0.0,
            'pitch'     : 0.0,
            'temperature': 0.0,
            'pressure'   : 0.0
        }

        self.log_fieldnames = [
            #Timestep
            'timestep',
            
            # Simulink Inputs (current timestep)
            'sim_simulation_time',
            'sim_dt_physics',
            'sim_battery_age_factor',
            'sim_battery_time_accel',
            'sim_speed',
            'sim_throttle',
            'sim_front_rpm',
            'sim_rear_rpm',
            'sim_range',
            'sim_battery_degradation',
            'sim_battery_soh',
            'sim_battery_temp',
            'sim_battery_temp_std',
            'sim_battery_soc',
            'sim_battery_curr',
            'sim_battery_cooling_temp',
            'carla_steering',
            'carla_roll',
            'carla_pitch',
            'carla_yaw',
            'carla_env_temp',
            'carla_traffic_light' #cheap check boolean for traffic lights 
        ]

        # Battery properties
        self.battery = {
            'max_capacity_kwh': 100.0,  # 100 kWh battery
            'current_soc': 100.0,       # State of Charge (%)
            'health': 100.0,            # Battery Health (%)
            'cycles': 0,                # Charge cycles
            'charging': False,
            'degradation_rate': 0.002,  # 0.2% degradation per cycle
            'recharge_threshold': 20.0, # Recharge at 20% SOC
            'charge_rate_kw': 50.0      # 50 kW charging
        }

        self.COLUMN_FORMATS = {
            # Simulink data
            'timestep': 4,
            'sim_speed': 2,
            'sim_throttle': 4,
            'sim_brake': 4,
            'sim_front_rpm': 1,
            'sim_front_enrgy': 2,
            'sim_rear_rpm': 1,
            'sim_rear_enrgy': 2,
            'sim_battery_temp': 2,
            'sim_battery_soc': 3,
            'sim_battery_curr': 2,
            'sim_battery_cooling_temp': 2,

            # Weather parameters
            'cloudiness': 1,
            'precipitation': 2,
            'wind_intensity': 2,
            'fog_density': 2,
            'wetness': 2,
            'sun_altitude': 1,
        }

        # Add shutdown flag and signal handler
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.carla_current_speed = 0.0
        self.carla_PID_reference_speed = 0.0

        print(f"Initialising Carla Environment")
        self._carla_init(carla_port, map_name)
        self.blueprints = self.world.get_blueprint_library()

        # Simulation init state
        print(f"Spawning Hero Car")
        self.hero_vehicle = self._spawn_hero_car()
        self.hero_vehicle.set_autopilot(True, 8000) # enable autopilot
        self.traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        self._update_weather()  # Initial weather setup
        #self.driver = CarlaLegacyDriver(self.world, self.hero_vehicle) #Create reference driver (basic PID driver, change out with automous model?)

        # Get blueprint library
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)
        traffic_manager.set_synchronous_mode(self.SYNC_MODE)

        # Set the vehicle to follow traffic rules
        traffic_manager.ignore_lights_percentage(self.hero_vehicle, 0.0)
        traffic_manager.ignore_signs_percentage(self.hero_vehicle, 0.0)
        traffic_manager.ignore_vehicles_percentage(self.hero_vehicle, 0.0)

        traffic_manager.auto_lane_change(self.hero_vehicle, True) # Avoid lane changes (optional)
        traffic_manager.distance_to_leading_vehicle(self.hero_vehicle, 2.0) # Keep a safe distance (2.0 m default)
        traffic_manager.vehicle_percentage_speed_difference(self.hero_vehicle, -5.0) #drive slightly faster cautious behavior (higher speed)

        self.traffic_manager = traffic_manager

        # initially spawn some NPCs
        self.npc_vehicles = self.spawn_npc_vehicles(self.init_vehicles)
        self.last_traffic_update = 0.0

        # TCP Bridge setup
        print(f"...Setup TCP connection for Simulink")
        self.network = self._setup_network(simulink_port)
        print(f"...Setup Data Logging")
        self.logging = self._setup_logging()
    
    def _carla_init(self, carla_port, map_name, retries=5, retry_delay=15.0):
        """Initialize CARLA connection with retry logic and validation"""
        for attempt in range(retries):
            try:
                # Initialize client
                self.client = carla.Client('localhost', carla_port)
                self.client.set_timeout(retry_delay)
                
                # Validate connection
                if not self.client.get_client_version() == "0.9.15":
                    raise RuntimeError("CARLA client/server version mismatch")
                
                # Load world with validation
                #current_map = self.client.get_world().get_map().name
                #if current_map != map_name:
                print(f"...Loading {map_name}")#(was {current_map})...
                self.world = self.client.load_world(map_name)
                #else:
                #    self.world = self.client.get_world()
                self.client.set_timeout(retry_delay)

                # Apply synchronous settings
                settings = self.world.get_settings()
                settings.synchronous_mode = self.SYNC_MODE
                settings.fixed_delta_seconds = 0.05 if self.SYNC_MODE else None
                
                self.client.set_timeout(retry_delay)

                # Validate settings application
                self.world.apply_settings(settings)
                applied_settings = self.world.get_settings()
                if self.SYNC_MODE and not applied_settings.synchronous_mode:
                    raise RuntimeError("Failed to enable synchronous mode")
                
                self.client.set_timeout(retry_delay)

                # Test world tick
                if self.SYNC_MODE:
                    self.world.tick()
                else:
                    self.world.wait_for_tick()
                
                print("CARLA initialized successfully")
                return True
                
            except Exception as e:
                print(f"CARLA init attempt {attempt+1} failed: {str(e)}")
                if attempt < retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Max retries reached. CARLA initialization failed.")
                    self._cleanup()
                    exit(1)

    def _spawn_hero_car(self):
        if self.SPAWN_HERO:
            hero_bp = self.client.get_world().get_blueprint_library().filter('model3')[0]
            hero_bp.set_attribute('role_name', 'hero')
            hero_transform = random.choice(self.world.get_map().get_spawn_points())
            hero_vehicle = self.world.try_spawn_actor(hero_bp, hero_transform)
            if hero_vehicle:
                print(f"Hero vehicle spawned at {hero_transform.location}.")
                return hero_vehicle
        return None

    def spawn_npc_vehicles(self, num):
        bpl = self.world.get_blueprint_library().filter("vehicle.*")
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        vehicles = []
        for i in range(min(num, len(spawn_points))):
            bp = random.choice(bpl)
            npc = self.world.try_spawn_actor(bp, spawn_points[i])
            if npc:
                npc.set_autopilot(True, self.traffic_manager.get_port())
                vehicles.append(npc)
        print(f"-> spawned {len(vehicles)} NPCs")
        return vehicles

    def _despawn_some(self, num_to_remove):
        to_remove = random.sample(self.npc_vehicles, min(num_to_remove, len(self.npc_vehicles)))
        for v in to_remove:
            try:
                v.destroy()
            except: pass
            self.npc_vehicles.remove(v)
        print(f"-> despawned {len(to_remove)} NPCs")

    def _setup_network(self, simulink_port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('localhost', simulink_port))
        self.sock.listen(1)
        self.sock.settimeout(60) # 20 seconds
        try:
            self.conn, _ = self.sock.accept()
        except socket.timeout:
            print("MATLAB failed to connect within timeout period.")
            self.shutdown_requested = True

        print("TCP setup completed")

    def _setup_logging(self):
        # Generate filename with preset info for easier identification
        filename = f'sim_{self.simulation_profile}_map{self.map_name}_soc{self.initSOC}_temp{self.CURRENT_TEMP:.0f}C_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        # Create data log file
        self.csv_file = open(filename, 'w', newline='')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.log_fieldnames)
        self.writer.writeheader()
        
        # Create/update metadata file
        metadata_file = 'simulation_metadata.csv'
        metadata_fieldnames = [
            'Simulation File', 
            'Timestamp', 
            'sim_profile',
            'Initial SOC', 
            'Ambient Temp (°C)',
            'Weather Preset',
            'Start Time'
        ]
        
        # Create metadata entry
        new_entry = {
            'Simulation File': filename,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sim_profile': self.simulation_profile,
            'Initial SOC': self.initSOC,
            'Ambient Temp (°C)': f"{self.CURRENT_TEMP:.1f}",
            'Weather Preset': self.selected_preset,
            'Start Time': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        # Write metadata (create file if needed)
        try:
            with open(metadata_file, 'a', newline='') as mfile:
                writer = csv.DictWriter(mfile, fieldnames=metadata_fieldnames)
                if mfile.tell() == 0:  # Write header only if file is empty
                    writer.writeheader()
                writer.writerow(new_entry)
        except Exception as e:
            print(f"Failed to write metadata: {str(e)}")
            raise

        print(f"Logging initialized: {filename}")
        return True
    
    def _update_weather(self):
        """Update weather while respecting preset constraints"""
        # Apply temperature change within preset bounds
        self.CURRENT_TEMP += np.random.uniform(-self.TEMP_CHANGE_RATE, self.TEMP_CHANGE_RATE)
        self.CURRENT_TEMP = np.clip(self.CURRENT_TEMP, self.MIN_TEMP, self.MAX_TEMP)
        
        # Force weather to match selected preset
        self.world.set_weather(self.selected_preset["weather"])
        self.current_weather = self.selected_preset["weather"]

    def _control_traffic_lights(self):
        for group in self.world.get_traffic_light_groups(): #Figure out how to get the traffic lights.
            for light in group:
                if light.state == carla.TrafficLightState.Red:
                    light.set_state(carla.TrafficLightState.Green)
                    break

    def _get_reference_speed_CARLA(self):
        """Get reference speed from legacy driver with safety checks"""
        if self.driver and self.hero_vehicle:
            try:
                return self.driver.get_reference_speed()
            except Exception as e:
                print(f"Error getting reference speed: {e}")
                return 0.0  # Fail-safe default
        return 0.0

    def _get_steering_control_CARLA(self):
        vehicle_control = self.hero_vehicle.get_control()
        steering = vehicle_control.steer  # Directly from autopilot
        return steering
    
    def _get_speed_control_SIMULINK(self):
        """
        Unused function
        """
        # Get motor RPM values from Simulink
        front_rpm = self.last_simulink_data['front_rpm']
        rear_rpm = self.last_simulink_data['rear_rpm']

        simulink_throttle = (front_rpm + rear_rpm) / self.MAX_POWER
        
        # Environment-aware speed adjustment
        current_speed = 3.6 * np.linalg.norm([
            self.hero_vehicle.get_velocity().x,
            self.hero_vehicle.get_velocity().y
        ])
        
        brake = 0.0
        #if self._check_traffic_light() or self._check_obstacles():
        #    brake = 1.0
        #    simulink_throttle = 0.0
            
        return simulink_throttle, brake

    def _check_traffic_light(self):
        return self.hero_vehicle.get_traffic_light_state() == carla.TrafficLightState.Red

    def _check_obstacles(self):
        hero_transform = self.hero_vehicle.get_transform()
        hero_forward = hero_transform.get_forward_vector()
        
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.id != self.hero_vehicle.id:
                delta = actor.get_location() - hero_transform.location
                distance = delta.length()
                if distance < 10.0 and delta.dot(hero_forward) > 0:
                    return True
        return False

    def _process_simulink_data(self):
        if not hasattr(self, 'recv_buffer'):
            self.recv_buffer = ""

        try:
            raw_data = self.conn.recv(2048).decode()#.strip()
            if not raw_data:
                print("MATLAB disconnected")
                self.shutdown_requested = True
                return
            
            if raw_data:
                if raw_data.count('{') != raw_data.count('}'):
                    print("Invalid JSON structure")
                    return
        
            self.recv_buffer += raw_data

            while '\n' in self.recv_buffer:
                line, self.recv_buffer = self.recv_buffer.split('\n', 1)
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        self.last_simulink_data.update(data)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
        
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Data error: {str(e)}")

    def _send_telemetry(self, throttle, reference_speed):
        vehicle_quartinion = self.hero_vehicle.get_transform().rotation

        self.telemetry = {
            'throttle': throttle,
            'speed' : reference_speed, #in km/h
            'roll'  : vehicle_quartinion.roll,
            'pitch' : vehicle_quartinion.pitch,
            'yaw'   : vehicle_quartinion.yaw,
            'temperature': self.CURRENT_TEMP,
            'pressure': self.atmPressureKPA,
        }

        payload = json.dumps(self.telemetry) + "\n"
        self.conn.sendall(payload.encode())
        
    def _format_numerical(self, entry):
        """Format all numerical values without scientific notation"""
        formatted = {}
        for key, value in entry.items():
            if key in self.COLUMN_FORMATS:
                try:
                    # Format as float with forced decimal notation
                    num = float(value)
                    fmt = f"%.{self.COLUMN_FORMATS[key]}f"
                    formatted[key] = float(fmt % num)
                except (ValueError, TypeError):
                    formatted[key] = value
            else:
                formatted[key] = value
        return formatted
    
    def parse_weather(weather_str):
        """Extracts key weather parameters from CARLA's string format"""
        params = dict(re.findall(r'(\w+)=([\d.]+)', weather_str))
    
        return {
            'ambient_temp': float(params.get('temperature', 0.0)),  # You need to track this separately!
            'rain_intensity': float(params['precipitation']),
            'road_wetness': float(params['precipitation_deposits']),
            'wind_speed': float(params['wind_intensity']),
            'solar_irradiance': (100 - float(params['cloudiness'])) / 100.0,  # Convert to 0-1 scale
        }

    def _log_data(self, snapshot, throttle, brake, steering, counter):
        # Capture current CARLA state directly
        rotation = self.hero_vehicle.get_transform().rotation

        # Formatting
        if self.last_simulink_data['vehicle_speed'] < np.finfo(np.float32).eps:
            self.last_simulink_data['vehicle_speed'] = 0.0
        
        log_entry = {
            #Timestep
            'timestep' : snapshot.timestamp.elapsed_seconds,
            
            # Simulink Inputs (current timestep)
            'sim_simulation_time': self.last_simulink_data['simulation_time'],
            'sim_dt_physics': self.last_simulink_data['dt_physics'],
            'sim_battery_age_factor': self.last_simulink_data['battery_age_factor'],
            'sim_battery_time_accel': self.last_simulink_data['time_battery_accel'],
            'sim_speed': self.last_simulink_data['vehicle_speed'],
            'sim_throttle': throttle,
            'sim_front_rpm': self.last_simulink_data['front_rpm'],
            'sim_rear_rpm': self.last_simulink_data['rear_rpm'],
            'sim_range': self.last_simulink_data['range'],
            'sim_battery_degradation': self.last_simulink_data['battery_degradation'],
            'sim_battery_soh': self.last_simulink_data['battery_soh'],
            'sim_battery_temp': self.last_simulink_data['battery_temp'],
            'sim_battery_temp_std' : self.last_simulink_data['battery_temp_std'],
            'sim_battery_soc': self.last_simulink_data['battery_soc'],
            'sim_battery_curr': self.last_simulink_data['battery_curr'],
            'sim_battery_cooling_temp': self.last_simulink_data['battery_cooling_temp'],

            # CARLA Outputs (current timestep)
            'carla_steering': steering,
            'carla_roll': rotation.roll,
            'carla_pitch': rotation.pitch,
            'carla_yaw': rotation.yaw,
            'carla_env_temp': self.CURRENT_TEMP,
            'carla_traffic_light': int(self._check_traffic_light()) #cheap check boolean for traffic lights 
        }

        # Remove the original carla_weather string field
        # formatted_entry = self._format_numerical(log_entry)
        self.writer.writerow(log_entry)
        if counter % 100 == 0:
            self.csv_file.flush()  # Ensure immediate write

    def _cleanup(self):
        """Enhanced cleanup with existence checks"""
        print("\nInitiating cleanup sequence...")
        
        # Close CSV file
        if hasattr(self, 'csv_file') and self.csv_file:
            try:
                self.csv_file.close()
                print("Data log closed")
            except Exception as e:
                print(f"Error closing log file: {str(e)}")

        # Network cleanup
        if hasattr(self, 'conn') and self.conn:
            try:
                self.conn.close()
                print("Simulink connection closed")
            except Exception as e:
                print(f"Error closing connection: {str(e)}")

        if hasattr(self, 'sock') and self.sock:
            try:
                self.sock.close()
                print("Network socket closed")
            except Exception as e:
                print(f"Error closing socket: {str(e)}")

        # CARLA cleanup
        if hasattr(self, 'hero_vehicle') and self.hero_vehicle:
            try:
                self.hero_vehicle.destroy()
                print("Hero vehicle destroyed")
            except Exception as e:
                print(f"Error destroying vehicle: {str(e)}")

        # Reset CARLA world settings
        if hasattr(self, 'world') and self.world:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                print("CARLA async mode restored")
            except Exception as e:
                print(f"Error resetting CARLA settings: {str(e)}")

        print("Cleanup completed successfully")
    
    def _handle_signal(self, signum):
        print(f"\nReceived shutdown signal {signum}, initiating cleanup...")
        self.shutdown_requested = True

    def run(self):
        print("Entering main loop")
        print(f"Sync mode:{self.SYNC_MODE}")
        counter = 0
        prev_action = None
        try:
            while not self.shutdown_requested:
                # Ensure Sync
                if self.SYNC_MODE:
                    self.world.tick()  # Crucial for synchronous mode
                    spectator = self.world.get_spectator()
                    transform = carla.Transform(self.hero_vehicle.get_transform().transform(carla.Location(x=-10, z=4)), self.hero_vehicle.get_transform().rotation)
                    spectator.set_transform(transform)
                    snapshot = self.world.get_snapshot()
                else:
                    snapshot = self.world.wait_for_tick()

                current_time = snapshot.timestamp.elapsed_seconds
                if current_time - self.last_temp_update > self.TEMP_UPDATE_INTERVAL:
                    self._update_weather()
                    self.last_temp_update = current_time

                # every UPDATE_INTERVAL seconds, vary traffic
                if current_time - self.last_traffic_update > self.veh_upd_interval:
                    self.last_traffic_update = current_time

                    # pick a new target size
                    target = random.randint(self.min_vehicles, self.max_vehicles)
                    current = len(self.npc_vehicles)
                    delta   = target - current

                    if delta > 0:
                        new_npcs = self.spawn_npc_vehicles(delta)
                        self.npc_vehicles.extend(new_npcs)
                    elif delta < 0:
                        self._despawn_some(-delta)

                # Process network data
                start_time = time.time()
                self._process_simulink_data() # fills curr = S_t , a_t-1
                elapsed = time.time() - start_time
                current_state  = self.last_simulink_data
                
                simulink_throttle  = current_state['throttle']      # = throttle_t
                steering_cmd  = self._get_steering_control_CARLA()  # steering_t
                action_cmd    = {'throttle': simulink_throttle,
                                'steering': steering_cmd} #steering

                if prev_action is not None:
                    steering_cmd  = self._get_steering_control_CARLA()  # steering_t
                    self._log_data(snapshot, action_cmd['throttle'], 0.0, steering_cmd, counter=counter)

                prev_action = action_cmd

                hero_velocity = self.hero_vehicle.get_velocity() #Get vehicle velocity vector
                reference_speed = np.linalg.norm([hero_velocity.x, hero_velocity.y, hero_velocity.z]) * 3.6 # Convert to km/h

                # Log and send data
                self._send_telemetry(simulink_throttle, reference_speed) #Send data from Carla to Matlab

                if counter % 10 == 0:
                    simulink_speed = self.last_simulink_data["vehicle_speed"]
                    print(f"Simulink loop time: {elapsed*1000:.2f}ms; SL throttle: {simulink_throttle:.4f}, SL Speed(km/h): {simulink_speed:.2f} Ref. speed(km/h): {reference_speed:.2f}, SL-RPM - Front: {self.last_simulink_data['front_rpm']:.2f}, Rear: {self.last_simulink_data['rear_rpm']:.2f}")
                counter += 1

        except Exception as e:
            print(f"Server error: {e}")
        
        finally:
            print("Exiting Simulation")
            self._cleanup()

if __name__ == '__main__':
    MAP_NAME = "Town01"
    INITIAL_VEHICLES = 40
    MIN_VEHICLES = 40
    MAX_VEHICLES = 100
    VEHICLE_UPDATE_INTERVAL = 30  # Traffic variation interval
    TRAFFIC_LIGHT_INTERVAL = 10  # Traffic light change interval
    TEMP_UPDATE_INTERVAL = 15  # Temperature change interval (seconds)

    sim = SimulinkCarlaCoSimulation(
        map_name=MAP_NAME, 
        init_vehicles=INITIAL_VEHICLES, 
        min_vehicles=MIN_VEHICLES, 
        max_vehicles=MAX_VEHICLES,
        veh_upd_interval = VEHICLE_UPDATE_INTERVAL,
        weather_preset_idx=2,
        soc_init=0.8, 
        simulation_profile="D5"
    )
    sim.run()