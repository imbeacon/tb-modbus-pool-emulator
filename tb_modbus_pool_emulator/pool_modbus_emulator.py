import asyncio
import logging
import random
from time import monotonic
from enum import Enum

from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore.store import ModbusSequentialDataBlock
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.server.async_io import StartAsyncTcpServer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Pool System State
class PoolSystemState:
    def __init__(self):
        # System configurations
        self.main_max_flow_rate = 200
        self.full_pool_volume = 55000  # liters

        # System states
        self.heating_system_enabled = True
        self.sand_filter_mode = SandFilterMode.ACTIVE_FILTRATION.value
        self.pump_running = True
        self.current_ph_level = 70
        self.target_ph_level = random.randint(50, 90)
        self.current_water_meter = 100.0
        self.pump_power_consumption = 50
        self.pump_rotation_speed = 1500
        self.filter_rotation_speed = random.randint(700, 2600)
        self.target_filter_rotation_speed = random.randint(700, 2600)
        self.heat_pump_power_consumption = 100
        self.heat_pump_rotation_speed = random.randint(50, 199)
        self.target_heat_pump_rotation_speed = random.randint(50, 200)
        self.low_water_level = False
        self.flow_rate = self.main_max_flow_rate
        self.weir_water_level = 90
        self.water_reaches_pool = True
        self.water_can_be_heated = True

        # Pressure initial values
        self.pressure = 6.0
        self.pump_pressure = 6.0
        self.current_pressure = 5.0
        self.pressure_units = 0

        # Temperature initial values
        self.target_temperature = 28.0
        self.ambient_temperature = random.uniform(15.0, 35.0)
        self.pump_temperature = self.ambient_temperature
        self.current_temperature = self.ambient_temperature
        self.current_in_temperature = self.current_temperature
        self.current_out_temperature = self.current_temperature
        self.compressor_temperature = 40.0

        # Valve initial states
        self.main_valve_opened = False
        self.pump_out_valve_opened = True
        self.drain_valve_opened = False
        self.pool_intake_valve_opened = True
        self.heating_outgoing_valve_opened = True
        self.heating_income_valve_opened = True
        self.throughpass_valve_opened = False
        self.pool_drain_valve_opened = True
        self.weir_valve_opened = True

        self.last_abnormal_vibration_update_time = 0

        self.calculate_water_reaches_pool()
        self.calculate_flow_rate()
        self.calculate_water_can_be_heated()

        self.devices = {}
        self.device_contexts = {}
        self.init_devices()

    def calculate_water_reaches_pool(self):
        self.water_reaches_pool = (self.pump_out_valve_opened and self.pool_intake_valve_opened
                                   and (self.throughpass_valve_opened
                                        or (self.heating_income_valve_opened and self.heating_outgoing_valve_opened)))

    def calculate_water_can_be_heated(self):
        self.water_can_be_heated = ((self.main_valve_opened
                                     or (self.pool_drain_valve_opened
                                         or (self.weir_valve_opened
                                             and self.current_water_meter > self.weir_water_level)))
                                    and self.pump_out_valve_opened and self.pool_intake_valve_opened
                                    and self.heating_income_valve_opened and self.heating_outgoing_valve_opened)

    # Function to update water level
    def update_water_level(self):
        water_flow_to_waste = (self.drain_valve_opened
                               and self.sand_filter_mode == SandFilterMode.WASTE.value
                               and self.pump_out_valve_opened
                               and (self.pool_drain_valve_opened
                                    or (self.weir_valve_opened and self.current_water_meter > self.weir_water_level)))

        water_level_change_rate = 0
        if self.main_valve_opened and self.water_reaches_pool:
            water_level_change_rate += 1
            if (water_flow_to_waste and not self.pool_drain_valve_opened
                    and not self.weir_valve_opened or self.current_water_meter < self.weir_water_level):
                water_level_change_rate += 1
        if water_flow_to_waste:
            water_level_change_rate -= 1

        water_level_change_rate -= 0.001  # Water level decreases due to evaporation
        self.current_water_meter = smooth_transition(self.current_water_meter,
                                                     max(0.0, min(100.0, self.current_water_meter +
                                                                  water_level_change_rate)),
                                                     step=abs(water_level_change_rate))

    # Function to calculate flow rate based on water level and pressure
    def calculate_flow_rate(self):
        water_has_intake = (self.main_valve_opened
                            or (self.pool_drain_valve_opened
                                or (self.weir_valve_opened and self.current_water_meter > self.weir_water_level)))
        max_flow_rate = self.main_max_flow_rate

        if not self.pump_running:
            if self.main_valve_opened:
                max_flow_rate = 50
            elif self.heating_system_enabled:
                max_flow_rate = 10
            else:
                max_flow_rate = 0

        if self.sand_filter_mode in [SandFilterMode.WASTE.value, SandFilterMode.CLOSED.value]:
            max_flow_rate //= 2
        else:
            max_flow_rate = max(60, max_flow_rate)

        pressure_factor = max(0.1, 1 - (self.pressure - 6) / 6) if self.pressure > 6 else max(0.01, self.pressure / 6)
        max_allowed_pressure_factor = 0.1 \
            if self.main_valve_opened and not self.pump_running else 0.1 if not self.pump_running else 1

        self.flow_rate = max(0, int(max_flow_rate * pressure_factor * max_allowed_pressure_factor)) \
            if water_has_intake else 0

    # Function to handle compressor temperature
    def handle_compressor_temperature(self):
        # Determine the step adjustment based on conditions
        if self.heating_system_enabled:
            if self.low_water_level or not self.water_can_be_heated:
                compressor_temperature_step_adjustment = 5.0 if self.pump_out_valve_opened else 10.0
            else:
                compressor_temperature_step_adjustment = 1.0
        else:
            compressor_temperature_step_adjustment = -1.0

        random_adjustment = random.randint(-5, 5)

        # Determine the maximum compressor temperature
        max_compressor_temp = 300.0 if self.low_water_level or not self.water_can_be_heated else random.randint(
            int(self.current_out_temperature * 10), 790 + random_adjustment) / 10

        # Smooth transition of compressor temperature based on the current state
        self.compressor_temperature = smooth_transition(
            self.compressor_temperature,
            max(
                min(max_compressor_temp, self.compressor_temperature + compressor_temperature_step_adjustment),
                self.ambient_temperature  # Ensure it doesn't drop below ambient temperature
            ),
            step=abs(compressor_temperature_step_adjustment)
        )

        # If the target temperature is less than or equal to the ambient temperature,
        # bring the compressor temperature down
        if self.target_temperature <= self.ambient_temperature and not self.heating_system_enabled:
            self.compressor_temperature = smooth_transition(self.compressor_temperature, self.ambient_temperature,
                                                            step=0.5)

        # If the compressor is actively heating (water can be heated and temperature is below target), slightly increase
        if self.heating_system_enabled and self.current_in_temperature < self.target_temperature - 1:
            self.compressor_temperature = smooth_transition(
                self.compressor_temperature,
                self.compressor_temperature + random.uniform(10.0, 15.0),
                step=abs(compressor_temperature_step_adjustment)
            )

    # Function to handle heat pump power consumption
    def handle_heat_pump_power_consumption(self):
        random_adjustment = random.randint(-5, 5)
        if self.heating_system_enabled and self.water_can_be_heated:
            if self.current_in_temperature < self.target_temperature:  # Heat pump is heating
                self.heat_pump_power_consumption = smooth_transition(self.heat_pump_power_consumption,
                                                                     min(300, self.heat_pump_power_consumption + 50 + random_adjustment),
                                                                     step=50)
            else:
                self.heat_pump_power_consumption = smooth_transition(self.heat_pump_power_consumption,
                                                                     max(200, self.heat_pump_power_consumption - 50 + random_adjustment),
                                                                     step=50)
        elif self.heating_system_enabled and not self.water_can_be_heated:
            # Water can't be heated, it doesn't reach to heat pump, but heat pump is still running and may overheat
            self.heat_pump_power_consumption = smooth_transition(self.heat_pump_power_consumption,
                                                                 min(3000, self.heat_pump_power_consumption + 50 + random_adjustment),
                                                                 step=50)

        else:
            self.heat_pump_power_consumption = 0

    # Function to handle pool temperatures
    def handle_pool_temperatures(self):
        current_in_temperature_adjustment = (self.ambient_temperature - self.current_in_temperature) * 0.01
        current_out_temperature_adjustment = (self.ambient_temperature - self.current_out_temperature) * 0.01
        flow_rate_factor = self.flow_rate / self.main_max_flow_rate
        if self.heating_system_enabled and self.water_can_be_heated:
            if self.heating_income_valve_opened and self.heating_outgoing_valve_opened:
                # Heating system is enabled and both valves are opened
                step_adjustment = 1
                if self.throughpass_valve_opened:  # Throughpass valve is opened
                    step_adjustment = 0.5
                if self.current_in_temperature < self.target_temperature - current_in_temperature_adjustment * 2:
                    self.current_out_temperature = max(self.ambient_temperature,
                                                       smooth_transition(self.current_out_temperature,
                                                                         (40 + (self.ambient_temperature - 40) * 0.1
                                                                          + current_out_temperature_adjustment),
                                                                         step=0.2 * step_adjustment * flow_rate_factor))
                else:
                    self.current_out_temperature = max(self.ambient_temperature,
                                                       smooth_transition(self.current_out_temperature,
                                                                         self.target_temperature
                                                                         + current_out_temperature_adjustment + 0.5,
                                                                         step=0.2 * step_adjustment * flow_rate_factor))
                if self.pool_intake_valve_opened:  # Pool intake valve is opened
                    self.current_in_temperature = max(self.ambient_temperature,
                                                      smooth_transition(self.current_in_temperature,
                                                                        self.target_temperature
                                                                        + current_in_temperature_adjustment,
                                                                        step=0.1 * step_adjustment * flow_rate_factor))
                else:  # Pool intake valve is closed so only ambient temperature is used
                    self.current_in_temperature = max(self.ambient_temperature,
                                                      smooth_transition(self.current_in_temperature,
                                                                        self.current_out_temperature
                                                                        + current_in_temperature_adjustment,
                                                                        step=0.2 * step_adjustment * flow_rate_factor))
                if self.current_out_temperature < self.current_in_temperature:
                    self.current_out_temperature = self.current_in_temperature
            else:  # Heating system is enabled but one of the valves is closed
                self.current_out_temperature = max(self.ambient_temperature,
                                                   smooth_transition(self.current_out_temperature,
                                                                     self.ambient_temperature
                                                                     + current_out_temperature_adjustment,
                                                                     step=0.2 * flow_rate_factor))
                self.current_in_temperature = max(self.ambient_temperature,
                                                  smooth_transition(self.current_in_temperature,
                                                                    self.ambient_temperature
                                                                    + current_in_temperature_adjustment,
                                                                    step=0.2 * flow_rate_factor))
        else:  # Heating system is disabled, pool slowly cools down
            self.current_out_temperature = max(self.ambient_temperature,
                                                  smooth_transition(self.current_out_temperature,
                                                                    self.ambient_temperature
                                                                    + current_out_temperature_adjustment,
                                                                    step=0.1 * flow_rate_factor))
            self.current_in_temperature = max(self.ambient_temperature,
                                                    smooth_transition(self.current_in_temperature,
                                                                        self.ambient_temperature
                                                                        + current_in_temperature_adjustment,
                                                                        step=0.1 * flow_rate_factor))

    def init_devices(self):
        self.devices = {
            "main_valve": create_device_store([0], [0] * 20, [0], [0] * 20),
            "pump": create_device_store([0], [0] * 20, [1], [0, 0, int(self.ambient_temperature * 10), 50, 1500]),
            "sand_filter": create_device_store([], [0] * 20, [], [0, 1, 95, 1000, 0, 0, 0] * 20),
            "throughpass_valve": create_device_store([0], [0] * 20, [0], [0] * 20),
            "heating_system": create_device_store([0], [0] * 20, [1],
                                                  [int(self.target_temperature * 10),
                                                   int(self.current_in_temperature * 10),
                                                   int(self.current_out_temperature * 10), 300,
                                                   int(self.ambient_temperature * 10), 100,
                                                   40, 1, 2]),
            "filter_ph_sensor": create_device_store([0], [70], [], [0] * 20),
            "heating_income_valve": create_device_store([0], [0] * 20, [1], [0] * 20),
            "heating_outgoing_valve": create_device_store([0], [0] * 20, [1], [0] * 20),
            "pool_intake_valve": create_device_store([0], [0] * 20, [1], [0] * 20),
            "weir_valve": create_device_store([0], [0] * 20, [1], [0] * 20),
            "pool_drain_valve": create_device_store([0], [0] * 20, [1], [0] * 20),
            "water_meter": create_device_store([0], [0] * 20, [0] * 20, [self.current_water_meter]),
            "pump_out_valve": create_device_store([0], [0] * 20, [1], [0] * 20),
            "drain_valve": create_device_store([0], [0] * 20, [1], [0] * 20),
        }

        # Define server contexts by port
        self.device_contexts = {
            5021: ModbusServerContext(slaves={1: self.devices["main_valve"]}, single=False),
            5022: ModbusServerContext(slaves={2: self.devices["pump"]}, single=False),
            5023: ModbusServerContext(slaves={3: self.devices["sand_filter"]}, single=False),
            5024: ModbusServerContext(slaves={4: self.devices["throughpass_valve"]}, single=False),
            5025: ModbusServerContext(slaves={5: self.devices["heating_system"]}, single=False),
            5026: ModbusServerContext(slaves={6: self.devices["filter_ph_sensor"]}, single=False),
            5027: ModbusServerContext(slaves={7: self.devices["heating_income_valve"]}, single=False),
            5028: ModbusServerContext(slaves={8: self.devices["heating_outgoing_valve"]}, single=False),
            5029: ModbusServerContext(slaves={9: self.devices["pool_intake_valve"]}, single=False),
            5030: ModbusServerContext(slaves={10: self.devices["weir_valve"]}, single=False),
            5031: ModbusServerContext(slaves={11: self.devices["pool_drain_valve"]}, single=False),
            5032: ModbusServerContext(slaves={12: self.devices["water_meter"]}, single=False),
            5033: ModbusServerContext(slaves={13: self.devices["pump_out_valve"]}, single=False),
            5034: ModbusServerContext(slaves={14: self.devices["drain_valve"]}, single=False),
        }

    # Function to handle pump logic
    def handle_pump_logic(self):
        random_adjustment = random.randint(-5, 5)
        pressure_random_adjustment = random.randint(-1, 1)
        pump_temperature_random_adjustment = random.randint(-7, 1)
        water_can_arrive_to_pump = (self.main_valve_opened or (self.pool_drain_valve_opened
                                         or (self.weir_valve_opened and self.current_water_meter > self.weir_water_level)))
        if self.pump_running and self.pump_out_valve_opened and water_can_arrive_to_pump:  # Pump is running and pump_out_valve is opened
            self.pump_pressure = smooth_transition(self.pump_pressure, min(19, self.pump_pressure + 0.5 + pressure_random_adjustment), step=0.2)
            self.pump_temperature = smooth_transition(self.pump_temperature, min(random.randint(60, 70), self.pump_temperature + 5 + pump_temperature_random_adjustment),
                                                      step=0.5)
            self.pump_power_consumption = smooth_transition(self.pump_power_consumption + random_adjustment,
                                                            random.randint(40, 160),
                                                            step=5)
            self.pump_rotation_speed = smooth_transition(self.pump_rotation_speed, random.randint(1000, 3000), step=100)
            self.current_pressure = smooth_transition(self.current_pressure, min(5, self.current_pressure + 0.5 + pressure_random_adjustment),
                                                      step=0.2)
        elif self.pump_running or (not water_can_arrive_to_pump and self.pump_running):  # Pump is running but pump_out_valve is closed
            self.pump_pressure = smooth_transition(self.pump_pressure, min(22, self.pump_pressure + 1.0 + pressure_random_adjustment), step=0.5)
            self.pump_temperature = smooth_transition(self.pump_temperature, min(100, self.pump_temperature + 5 + pump_temperature_random_adjustment),
                                                      step=0.5)
            self.pump_power_consumption = smooth_transition(self.pump_power_consumption + random_adjustment,
                                                            min(300, self.pump_power_consumption + 50),
                                                            step=50)
            self.pump_rotation_speed = smooth_transition(self.pump_rotation_speed, random.randint(1000, 3000), step=100)
            self.current_pressure = 1  # Pressure in sand filter is 1 atm if pump_out_valve is closed
        elif self.pump_out_valve_opened:  # Pump is not running but pump_out_valve is opened
            self.pump_pressure = smooth_transition(self.pump_pressure, max(1, self.pump_pressure - 0.3 + pressure_random_adjustment), step=0.2)
            self.pump_temperature = smooth_transition(self.pump_temperature,
                                                      max(self.ambient_temperature, self.pump_temperature - 1 + pump_temperature_random_adjustment), step=1)
            self.pump_power_consumption = 0
            self.pump_rotation_speed = 0
            self.current_pressure = smooth_transition(self.current_pressure, max(1, self.current_pressure - 0.3 + pressure_random_adjustment),
                                                      step=0.2)
        else:  # Pump is off and pump_out_valve is closed
            self.pump_pressure = smooth_transition(self.pump_pressure, max(1, self.pump_pressure - 0.2 + pressure_random_adjustment), step=0.2)
            self.pump_temperature = smooth_transition(self.pump_temperature,
                                                      max(self.ambient_temperature, self.pump_temperature - 0.5 + pump_temperature_random_adjustment),
                                                      step=0.5)
            self.pump_power_consumption = 0
            self.pump_rotation_speed = 0
            self.current_pressure = 1

    def update_random_values(self):
        # Update pH level
        self.target_ph_level = random.randint(50, 90)
        self.current_ph_level = smooth_transition(self.current_ph_level, self.target_ph_level, step=1)
        self.devices["filter_ph_sensor"].setValues(6, 0, [int(self.current_ph_level)])

        # Update filter rotation speed
        if self.filter_rotation_speed == self.target_filter_rotation_speed:
            self.target_filter_rotation_speed = random.randint(700, 2600)
        filter_rotation_speed = smooth_transition(self.filter_rotation_speed, self.target_filter_rotation_speed,
                                                  step=10)
        self.devices["sand_filter"].setValues(6, 3, [int(filter_rotation_speed)])
        self.devices["sand_filter"].setValues(6, 6, [random.randint(0, 2) if self.sand_filter_mode != SandFilterMode.CLOSED.value else 0])

        # Update heat pump rotation speed
        if self.target_heat_pump_rotation_speed - 10 < self.heat_pump_rotation_speed < self.target_heat_pump_rotation_speed + 10:
            self.target_heat_pump_rotation_speed = random.randint(50, 199)
        heat_pump_rotation_speed = smooth_transition(self.heat_pump_rotation_speed,
                                                     self.target_heat_pump_rotation_speed, step=5)
        self.devices["heating_system"].setValues(6, 5, [int(heat_pump_rotation_speed) if self.heating_system_enabled else 0])
        self.devices["heating_system"].setValues(6, 8, [random.randint(2, 3) if self.heating_system_enabled else 0]) # refrigerant pressure
        self.devices["heating_system"].setValues(6, 4, [int(self.ambient_temperature * 10 + random.randint(-5, 5))])
        if self.last_abnormal_vibration_update_time == 0 or monotonic() - self.last_abnormal_vibration_update_time > 10:
            self.devices["heating_system"].setValues(6, 7, [random.randint(0, 2) if self.heating_system_enabled else 0])
            self.devices["pump"].setValues(6, 2, [random.randint(0, 2) if self.pump_running else 0])

    def get_devices_state(self):
        self.main_valve_opened = self.devices["main_valve"].getValues(1, 0, count=1)[0]
        self.pump_running = self.devices["pump"].getValues(1, 0, count=1)[0]
        self.pump_out_valve_opened = self.devices["pump_out_valve"].getValues(1, 0, count=1)[0]
        self.sand_filter_mode = self.devices["sand_filter"].getValues(3, 1, count=1)[0]
        self.pressure_units = self.devices["sand_filter"].getValues(3, 4, count=1)[0]
        self.filter_rotation_speed = self.devices["sand_filter"].getValues(3, 3, count=1)[0]
        self.pool_drain_valve_opened = self.devices["pool_drain_valve"].getValues(1, 0, count=1)[0]
        self.drain_valve_opened = self.devices["drain_valve"].getValues(1, 0, count=1)[0]
        self.throughpass_valve_opened = self.devices["throughpass_valve"].getValues(1, 0, count=1)[0]
        self.heating_income_valve_opened = self.devices["heating_income_valve"].getValues(1, 0, count=1)[0]
        self.heating_outgoing_valve_opened = self.devices["heating_outgoing_valve"].getValues(1, 0, count=1)[0]
        self.pool_intake_valve_opened = self.devices["pool_intake_valve"].getValues(1, 0, count=1)[0]
        self.weir_valve_opened = self.devices["weir_valve"].getValues(1, 0, count=1)[0]
        self.heating_system_enabled = self.devices["heating_system"].getValues(1, 0, count=1)[0]
        self.target_temperature = self.devices["heating_system"].getValues(3, 0, count=1)[0] / 10
        self.heat_pump_rotation_speed = self.devices["heating_system"].getValues(3, 5, count=1)[0]

        if self.sand_filter_mode == SandFilterMode.WASTE.value and not self.drain_valve_opened:
            self.drain_valve_opened = True
            self.devices["drain_valve"].setValues(1, 0, [1])

    def update_device_values(self, adjusted_pressure):
        flow_rate_to_save = max(0, int(self.flow_rate + random.randint(-10, 10) - 10))
        self.devices["water_meter"].setValues(6, 0, [int(self.current_water_meter)])  # Update water level
        self.devices["pump"].setValues(6, 1, [flow_rate_to_save])  # Update flow rate in L/min
        self.devices["pump"].setValues(6, 3, [int(self.pump_temperature * 10)])  # Update temperature in °C
        self.devices["pump"].setValues(6, 4, [int(self.pump_power_consumption)])  # Update power consumption in W
        self.devices["pump"].setValues(6, 5, [int(self.pump_rotation_speed)])  # Update rotation speed
        self.devices["pump"].setValues(6, 6, [int(self.pump_pressure)])  # Update pump pressure
        self.devices["sand_filter"].setValues(6, 0, [flow_rate_to_save])  # Update flow rate in sand filter
        self.devices["sand_filter"].setValues(6, 1, [int(self.sand_filter_mode)])  # Update state in sand filter
        self.devices["sand_filter"].setValues(6, 5, [int(adjusted_pressure)])  # Update pressure in selected units
        self.devices["heating_system"].setValues(6, 1,
                                                 [int(self.current_in_temperature * 10)])  # Update currentInTemperature
        self.devices["heating_system"].setValues(6, 2, [
            int(self.current_out_temperature * 10)])  # Update currentOutTemperature
        self.devices["heating_system"].setValues(6, 6, [
            int(self.compressor_temperature * 10)])  # Update compressor temperature
        self.devices["heating_system"].setValues(6, 3, [int(self.heat_pump_power_consumption)])  # Update heat pump power consumption

        self.update_random_values()


# Enum for Sand Filter Modes
class SandFilterMode(Enum):
    ACTIVE_FILTRATION = 1
    WASTE = 2
    BACKWASH = 3
    RECIRCULATE = 4
    RINSE = 5
    CLOSED = 6


# Function to create a Modbus device store
def create_device_store(digital_inputs=None, input_registers=None, coils=None, holding_registers=None):
    return ModbusSlaveContext(
        di=ModbusSequentialDataBlock(1, create_initial_values(digital_inputs)),
        co=ModbusSequentialDataBlock(1, create_initial_values(coils)),
        hr=ModbusSequentialDataBlock(1, create_initial_values(holding_registers)),
        ir=ModbusSequentialDataBlock(1, create_initial_values(input_registers))
    )


def create_initial_values(initial_values):
    return (initial_values or []) + [0] * (16 - len(initial_values or []))


# Function to smoothly transition values
def smooth_transition(current, target, step=0.1):
    diff = abs(target - current)
    adaptive_step = max(step, diff * 0.1)
    return min(current + adaptive_step, target) if current < target else max(current - adaptive_step, target)


# Function to convert pressure units
def convert_pressure(pressure, units):
    return pressure * 100 if units == 1 else pressure


# Function to update the device values periodically and handle RPCs
async def update_values(pool_system_state, running_event):
    while running_event.is_set():
        try:
            # Get device states
            pool_system_state.get_devices_state()

            pool_system_state.low_water_level = pool_system_state.current_water_meter < 20

            # Handle pump logic
            pool_system_state.handle_pump_logic()
            pool_system_state.calculate_water_reaches_pool()
            pool_system_state.update_water_level()
            pool_system_state.calculate_water_can_be_heated()

            # Handle compressor temperature
            pool_system_state.handle_compressor_temperature()

            # Handle heat pump power consumption
            pool_system_state.handle_heat_pump_power_consumption()

            # Adjust temperatures
            pool_system_state.handle_pool_temperatures()

            # Convert pressure to the selected units
            adjusted_pressure = convert_pressure(pool_system_state.current_pressure, pool_system_state.pressure_units)

            # Calculate flow rate
            pool_system_state.calculate_flow_rate()

            # Update telemetry values
            pool_system_state.update_device_values(adjusted_pressure)

            # Log updated parameters
            logging.info(
                f"Main Valve Opened: {pool_system_state.main_valve_opened}, "
                f"Pump Running: {pool_system_state.pump_running}, "
                f"Pump Out Valve Opened: {pool_system_state.pump_out_valve_opened}")
            logging.info(
                f"Pump Temperature: {pool_system_state.pump_temperature}, "
                f"Pump Power Consumption: {pool_system_state.pump_power_consumption}, "
                f"Pump Rotation Speed: {pool_system_state.pump_rotation_speed}")
            logging.info(
                f"Compressor Temperature: {pool_system_state.compressor_temperature}, "
                f"Flow Rate: {pool_system_state.flow_rate}, "
                f"Water Level: {pool_system_state.current_water_meter}")
            logging.info(
                f"Sand Filter Mode: {pool_system_state.sand_filter_mode}, "
                f"Filter Rotation Speed: {pool_system_state.filter_rotation_speed}, "
                f"Pressure: {adjusted_pressure}, "
                f"Pump Pressure: {pool_system_state.pump_pressure}")
            logging.info(
                f"Current In Temperature: {pool_system_state.current_in_temperature}, "
                f"Current Out Temperature: {pool_system_state.current_out_temperature}")
            logging.info(f"Target temperature: {pool_system_state.target_temperature}, "
                         f"Ambient temperature: {pool_system_state.ambient_temperature}")
            logging.info(
                f"Heating System Enabled: {pool_system_state.heating_system_enabled}, "
                f"Low Water Level: {pool_system_state.low_water_level}, "
                f"Pool Drain Valve Opened: {pool_system_state.pool_drain_valve_opened}")
            logging.info(
                f"Drain Valve Opened: {pool_system_state.drain_valve_opened}, "
                f"Throughpass Valve Opened: {pool_system_state.throughpass_valve_opened}, "
                f"Heating Income Valve Opened: {pool_system_state.heating_income_valve_opened}")
            logging.info(
                f"Heating Outgoing Valve Opened: {pool_system_state.heating_outgoing_valve_opened}, "
                f"Pool Intake Valve Opened: {pool_system_state.pool_intake_valve_opened}, "
                f"Weir Valve Opened: {pool_system_state.weir_valve_opened}")
            logging.info("Heating System power consumption: " + str(pool_system_state.heat_pump_power_consumption))
            logging.info("Heating System rotation speed: " + str(pool_system_state.heat_pump_rotation_speed))
            logging.info("Sand filter vibration: " + str(pool_system_state.devices["sand_filter"].getValues(6, 6, count=1)[0]))
            logging.info("")

            await asyncio.sleep(2)

        except Exception as e:
            logging.error(f"Error during update: {str(e)}")
            await asyncio.sleep(2)


# Function to set pump vibration to trigger alarm
async def set_pump_vibration(pool_system_state):
    while True:
        await asyncio.sleep(1800 + random.randint(-600, 600))
        if pool_system_state.pump_running:
            pool_system_state.last_abnormal_vibration_update_time = monotonic()
            pool_system_state.devices["pump"].setValues(6, 2, [random.uniform(5, 7)])

# Function to set heat pump vibration to trigger alarm
async def set_heat_pump_vibration(pool_system_state):
    while True:
        await asyncio.sleep(1800 + random.randint(-600, 600))
        if pool_system_state.heating_system_enabled:
            pool_system_state.last_abnormal_vibration_update_time = monotonic()
            pool_system_state.devices["heating_system"].setValues(6, 2, [random.uniform(5, 7)])

# Function to start each server
async def start_server(port, context):
    logging.info(f"Starting Modbus server on port {port}")
    await StartAsyncTcpServer(context=context, identity=identity, address=("0.0.0.0", port))


# Start the Modbus TCP servers for each device
async def run_servers(pool_system_state):
    server_tasks = [start_server(port, context) for port, context in pool_system_state.device_contexts.items()]
    await asyncio.gather(*server_tasks)


# Define server identity
identity = ModbusDeviceIdentification()
identity.VendorName = 'ThingsBoard'
identity.ProductCode = 'PoolSystemSimulator'
identity.VendorUrl = 'https://thingsboard.io/'
identity.ProductName = 'Pool System Emulator'
identity.ModelName = 'Modbus TCP Emulator'
identity.MajorMinorRevision = '1.0'

# Run the servers and value updates
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    running = asyncio.Event()
    running.set()
    state = PoolSystemState()
    loop.create_task(run_servers(state))
    loop.create_task(update_values(state, running))
    loop.create_task(set_pump_vibration(state))
    loop.create_task(set_heat_pump_vibration(state))
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        running.clear()
        logging.info("Emulator stopped")
